import os
import json
import torch
import argparse
import transformers
from tqdm import tqdm
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run paraphrasing attack against blanked input using a specified model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the HuggingFace model.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing {algorithm}_blank.json files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to write {algorithm}_attack.json files.')
    parser.add_argument('--algorithms', type=str, default='KGW,Unigram,UPV,DIP,EWD,EXP,SIR',
                        help='Comma-separated list of watermark algorithms to process.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU IDs to use, e.g., "0,1,2"')
    return parser.parse_args()


def fill_attack_prompt(reference_text, blank_text):
    prompt = (
        "You will be shown one reference paragraph and one incomplete paragraph.\n"
        "Your task is to write a complete paragraph using incomplete paragraph.\n"
        "The complete paragraph should have similar length with reference paragraph.\n"
        "You need to include all the information in the reference. \n"
        "But do not take the expression and words in the reference paragraph.\n"
        "You should only answer the complete paragraph.\n"
        f"reference: {reference_text}\n"
        f"incomplete pragraph: {blank_text}\n"
    )
    return prompt


def main(args):
    print(f"PID: {os.getpid()}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load model and build generation pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        do_sample=False
    )

    algorithms = args.algorithms.split(",")

    for algorithm in algorithms:
        input_file = os.path.join(args.input_dir, f'{algorithm}_blank.json')
        output_file = os.path.join(args.output_dir, f'{algorithm}_attack.json')

        if not os.path.exists(input_file):
            print(f"[!] Skipping {algorithm}: input file not found → {input_file}")
            continue

        with open(input_file, 'r') as f:
            lines = f.readlines()

        last_line = 0
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                last_line = sum(1 for _ in f)

        with open(output_file, 'w') as out_f:
            for line in tqdm(lines[last_line:], desc=f"Processing {algorithm}", unit="line"):
                item = json.loads(line)
                prompt = item['prompt']
                watermarked_text = item['watermarked_text']
                unwatermarked_text = item['unwatermarked_text']
                blank_text = item['blank_text']
                ref_text = item['ref_text']

                input_text = fill_attack_prompt(ref_text, blank_text)
                messages = [{"role": "user", "content": input_text}]
                outputs = pipeline(messages, max_new_tokens=256, do_sample=False)

                output_text = outputs[0]["generated_text"][-1]["content"]

                response_item = {
                    'prompt': prompt,
                    'watermarked_text': watermarked_text,
                    'unwatermarked_text': unwatermarked_text,
                    'blank_text': blank_text,
                    'ref_text': ref_text,
                    'attack_text': output_text,
                }
                out_f.write(json.dumps(response_item) + '\n')

    del pipeline
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    main(args)
