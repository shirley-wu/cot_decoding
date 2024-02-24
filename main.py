#!/usr/bin/env python
# coding=utf-8

import json
import os
from dataclasses import field, dataclass
from functools import partial

import numpy as np
import torch
import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, HfArgumentParser

from solve import DecodingArguments, solve
from task import GSMTask


@dataclass
class MainArguments:
    data_file: str = field(
        default="./gsm8k_data/test.jsonl"
    )
    model_name_or_path: str = field(
        default="mistralai/Mistral-7B-Instruct-v0.1"
    )  # `mistralai/Mistral-7B-Instruct-v0.1` or `mistralai/Mistral-7B-v0.1`
    batch_size: int = field(
        default=64
    )
    output_fname: str = field(
        default="outputs/model_predictions.jsonl"
    )


def encode_function(example, tokenizer, task):
    prompt = task.encode_prompt(example)
    tokenized = tokenizer(prompt, return_tensors='pt')
    input_ids = tokenized.input_ids
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


@torch.no_grad()
def main():
    parser = HfArgumentParser((MainArguments, DecodingArguments))
    main_args, decoding_args = parser.parse_args_into_dataclasses()
    task = GSMTask(encode_format=decoding_args.encode_format)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(main_args.model_name_or_path, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        main_args.model_name_or_path, torch_dtype=torch.bfloat16, device_map='auto'
    )

    # Load dataset
    raw_dataset = load_dataset("json", data_files={'test': main_args.data_file})['test']
    encode_function_partial = partial(
        encode_function,
        tokenizer=tokenizer,
        task=task,
    )
    lm_dataset = raw_dataset.map(
        encode_function_partial,
        batched=False,
        num_proc=16,
        remove_columns=[name for name in raw_dataset.column_names if name not in ["input_ids", "attention_mask"]],
        desc="Tokenizing data",
    )

    # Generate
    dataloader = DataLoader(
        lm_dataset, shuffle=False, batch_size=main_args.batch_size,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
    )
    outputs_all = []
    accs_all = []
    pbar = tqdm.tqdm(dataloader, total=len(dataloader))
    for batch in pbar:
        outputs = solve(model, tokenizer, task, batch, args=decoding_args)
        n = len(outputs_all)
        for i, output in enumerate(outputs):
            output['is_correct'] = task.is_correct(raw_dataset[n + i], output['answer'])
            outputs_all.append(output)
            accs_all.append(output['is_correct'])
        pbar.set_postfix(acc=np.mean(accs_all))

    # Evaluate & dump results
    print("Acc = %.2f" % (np.mean(accs_all) * 100))
    output_fname = main_args.output_fname
    while os.path.exists(output_fname):
        opt = None
        while opt not in ['y', 'n', ]:
            opt = input("{} exists. Do you want to overwrite? [y/n] - ")
        if opt == 'y':
            break
        else:
            output_fname = input("Input a new filename: ")
    if output_fname != main_args.output_fname:
        print("OK, will save to", output_fname)
    os.makedirs(os.path.dirname(main_args.output_fname), exist_ok=True)
    with open(main_args.output_fname, "w") as f:
        for output in outputs_all:
            f.write(json.dumps(output) + '\n')


if __name__ == "__main__":
    main()
