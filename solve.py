import collections
from dataclasses import dataclass, field

import torch


@dataclass
class DecodingArguments:
    encode_format: str = field(
        default="instruct",  # choices=["instruct", "qa", ]
    )  # choices: instrct / qa
    max_new_tokens: int = field(
        default=512,
    )  # 512 for `mistralai/Mistral-7B-Instruct-v0.1`, 256 for `mistralai/Mistral-7B-v0.1`
    decoding: str = field(
        default="greedy",  # choices=["greedy", "cot", ],
    )
    cot_n_branches: int = field(default=10)
    cot_aggregate: str = field(
        default="sum"  # choices=["max", "sum", "self_consistency", ]
    )


def greedy_decoding_solve(model, tokenizer, task, batch, args: DecodingArguments):
    gen_ids = model.generate(
        input_ids=batch.input_ids.cuda(), attention_mask=batch.attention_mask.cuda(),
        do_sample=False, max_new_tokens=args.max_new_tokens,
    )
    ret = []
    for i in range(len(gen_ids)):
        text = tokenizer.decode(gen_ids[i, batch.input_ids.shape[-1]:], skip_special_tokens=True)
        answer, answer_span = task.extract_model_answer(text)
        ret.append({'text': text, 'answer': answer, 'answer_span': answer_span})
    return ret


def cot_decoding_solve(model, tokenizer, task, batch, args: DecodingArguments):
    bsz = batch.input_ids.shape[0]
    n_branches = args.cot_n_branches

    input_ids = model.generate(
        input_ids=batch.input_ids.cuda(), attention_mask=batch.attention_mask.cuda(),
        do_sample=False, num_beams=n_branches, num_return_sequences=n_branches, max_new_tokens=1,
        min_new_tokens=1,  # don't generate eos, it's stupid
    )
    attention_mask = batch.attention_mask.cuda().repeat_interleave(n_branches, 0)
    attention_mask = torch.cat([
        attention_mask, torch.ones((len(attention_mask), 1), device=attention_mask.device, dtype=attention_mask.dtype),
    ], dim=1)

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, do_sample=False, max_new_tokens=args.max_new_tokens - 1,
        output_logits=True, return_dict_in_generate=True,
    )
    gen_ids = outputs['sequences'][:, input_ids.shape[1] - 1:].reshape(bsz, n_branches, -1)
    gen_probs = torch.stack(outputs['logits'], dim=1).softmax(-1)
    n_vocab = gen_probs.shape[-1]
    gen_probs = torch.cat(
        [torch.full((bsz * n_branches, 1, n_vocab), 1 / n_vocab, dtype=gen_probs.dtype, device=gen_probs.device),
         gen_probs, ], dim=1)  # concat a pseudo probability for first token
    gen_probs = gen_probs.reshape(bsz, n_branches, -1, n_vocab)

    def decode_with_offsets(generation_ids):
        # I'm not aware of any more convenient way to do this. If you know, please do let me know
        tokens = tokenizer.convert_ids_to_tokens(generation_ids)

        text = ''
        offsets = []
        for i in range(len(generation_ids)):
            if tokens[i] == tokenizer.eos_token:
                break
            text = tokenizer.convert_tokens_to_string(tokens[:i + 1])
            offsets.append(len(text))
        offsets += [-1 for _ in range(len(tokens) - len(offsets))]  # add invalid offsets for EOS

        return text, offsets

    def match_answer_span(answer_span, offsets):
        answer_s, answer_e = answer_span
        inds = []
        for i, offset in enumerate(offsets):
            if answer_s < offset:
                inds.append(i)
                if answer_e <= offset:
                    break
        return inds

    def get_cot_score(probs):
        probs = probs.topk(k=2, dim=-1, sorted=True).values
        score = (probs[:, 0] - probs[:, 1]).mean()
        return float(score)

    ret = []
    for i in range(bsz):
        # Sample candidates
        candidates = []
        for j in range(n_branches):
            text, offsets = decode_with_offsets(gen_ids[i, j])
            answer, answer_span = task.extract_model_answer(text)
            if answer_span is None:
                candidates.append({'text': text, 'answer': answer, 'answer_span': answer_span, 'score': 0})
            else:
                answer_tokens = match_answer_span(answer_span, offsets)
                answer_probs = gen_probs[i, j][torch.LongTensor(answer_tokens).cuda()]
                cot_score = get_cot_score(answer_probs)
                candidates.append({'text': text, 'answer': answer, 'answer_span': answer_span, 'score': cot_score})
        ret.append({'candidates': candidates})

        # Aggregate candidates
        if args.cot_aggregate == 'max':
            answer = sorted(candidates, key=lambda x: x['score'], reverse=True)[0]['answer']
            ret[-1]['answer'] = answer
        elif args.cot_aggregate == 'sum':
            answer_scores = {}
            for candidate in candidates:
                answer = candidate['answer']
                answer_scores[answer] = answer_scores.get(answer, 0) + candidate['score']
            answer = sorted(answer_scores.items(), key=lambda x: x[1], reverse=True)[0][0]
            ret[-1]['answer_scores'] = answer_scores
            ret[-1]['answer'] = answer
        else:
            assert args.cot_aggregate == 'self_consistency'  # try a straight-forward but strong baseline
            counter = collections.Counter([cand['answer'] for cand in candidates])
            answer = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
            ret[-1]['answer'] = answer

    return ret


def solve(model, tokenizer, task, batch, args: DecodingArguments):
    if args.decoding == 'greedy':
        return greedy_decoding_solve(model, tokenizer, task, batch, args)
    elif args.decoding == 'cot':
        return cot_decoding_solve(model, tokenizer, task, batch, args)
    else:
        raise ValueError("Invalid decoding " + args.decoding)
