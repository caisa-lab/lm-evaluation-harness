from lm_eval import tasks, evaluator, utils
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model_name", dest="model_name", required=True, type=str)
parser.add_argument("--tasks", dest="tasks", required=True, type=str,)
parser.add_argument("--write_out", action="store_true", default=False)
parser.add_argument("--output_base_path", type=str, default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(f"{model_name}", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True) # torch_dtype=torch.float16, device_map="auto"
    model_name = model_name.split('/')[-1]
    tasks = [t.strip() for t in args.tasks.split(",")]
    print("Evaluating on the following tasks:")
    print(tasks)
    results = evaluator.simple_evaluate(
        model=model,
        # model_args=model,
        tasks=tasks,
        num_fewshot=0,
        batch_size=4,
        max_batch_size=4,
        device='cuda:0', #
        # no_cache=args.no_cache,
        # limit=args.limit,
        # description_dict=description_dict,
        # decontamination_ngrams_path=args.decontamination_ngrams_path,
        # check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    output_path = f'data/llm_eval/crows_pairs_english_french_breakdown_{model_name}.json'
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(dumped)
