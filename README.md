A personal reproduction of DeepMind's recent great work: [**Chain-of-Thought Reasoning Without Prompting**](https://arxiv.org/pdf/2402.10200.pdf)

## Reproduction Results

Results with `mistralai/Mistral-7B-Instruct-v0.1` and `mistralai/Mistral-7B-v0.1` on GSM-8k dataset:

|                         | Mistral-base | Mistral-Instruct |
|-------------------------|--------------|------------------|
| Greedy                  | 10.24        | 32.22            |
| Self consistency        | 14.63        | 46.02            |
| CoT decoding (agg path) | **20.17**    | **46.40**        |
| CoT decoding (max path) | 13.72        | 39.27            |

For comparison, the results reported in the original paper are:

|                         | Mistral-base | Mistral-Instruct |
|-------------------------|--------------|------------------|
| Greedy                  | 9.9          | 31.2             |
| CoT decoding (agg path) | **25.1**     | **38.2**         |

## How to Run

```bash
python main.py --model_name_or_path mistralai/Mistral-7B-Instruct-v0.1 --encode_format instruct --max_new_tokens 512 --decoding cot --output_fname outputs/mistral-instruct.jsonl
python main.py --model_name_or_path mistralai/Mistral-7B-v0.1 --encode_format qa --max_new_tokens 256 --decoding cot --output_fname outputs/mistral-base.jsonl
```

Please adjust batch size by `--batch_size xxx` based on your own GPU configuraion.

## Dependency

Install `transformers==4.38.1`. It's crucial to have >=4.38.0!