import re


class GSMTask:
    # partially adapted from https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py

    GT_ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    INVALID_ANS = "[invalid]"
    MODEL_ANS_RE = re.compile(r"([-0-9][0-9\,\.]*[0-9])|([0-9])")

    def __init__(self, encode_format):
        self.encode_format = encode_format
        assert self.encode_format in ['instruct', 'qa', ]

    def encode_prompt(self, example):
        if self.encode_format == 'instruct':
            return '[INST]{}[/INST]'.format(example['question'])
        elif self.encode_format == 'qa':
            return 'Q: {}\nA:'.format(example['question'])

    def extract_gt_answer(self, completion):
        match = self.GT_ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return self.INVALID_ANS

    def extract_model_answer(self, completion):
        if self.encode_format == 'qa':
            completion = completion.split("\nQ: ")[0]

        matches = list(re.finditer(self.MODEL_ANS_RE, completion))
        if len(matches) > 0:
            match = matches[-1]
            return match.group(), (match.start(), match.end())
        else:
            return self.INVALID_ANS, None

    def is_correct(self, gt_example, model_answer):
        gt_answer = self.extract_gt_answer(gt_example["answer"])
        assert gt_answer != self.INVALID_ANS
        return model_answer == gt_answer
