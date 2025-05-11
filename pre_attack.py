import os
import json
import torch
import numpy as np
from tqdm import tqdm
import argparse
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from visualize.font_settings import FontSettings
from visualize.visualizer import ContinuousVisualizer
from visualize.legend_settings import ContinuousLegendSettings
from visualize.page_layout_settings import PageLayoutSettings
from visualize.color_scheme import ColorSchemeForContinuousVisualization

def parse_args():
    parser = argparse.ArgumentParser(description="Run paraphrasing + self-information blanking pipeline.")
    parser.add_argument('--input_dir', type=str, default='/root/Self-information-Rewrite-Attack-main/dataset/c4/watermarked/',
                        help='Directory containing {algorithm}_response.json input files')
    parser.add_argument('--result_dir', type=str, default='/root/Self-information-Rewrite-Attack-main/dataset/c4/watermarked/',
                        help='Directory to store ref and blank output files')
    parser.add_argument('--model_path', type=str, default='/root/models/meta-llama/Llama-3.2-3B-Instruct/')  
    parser.add_argument('--threshold', type=int, default=30)  
    parser.add_argument('--gpu', type=str, default='0', help='GPU IDs to use, e.g., "0,1,2"')
    parser.add_argument('--algorithms', type=str, default='KGW',
                        help='Comma-separated list of watermark algorithms to process')
    return parser.parse_args()

def fill_parapharseprompt(input_text):
    return (
        "You are a paraphraser. You are given an input passage 'INPUT'. "
        "You should paraphrase 'INPUT' to print 'OUTPUT'. 'OUTPUT' should be diverse and different "
        "as much as possible from 'INPUT' and should not copy any part verbatim from 'INPUT'. "
        "However, 'OUTPUT' should preserve the information in the INPUT. "
        "You should print 'OUTPUT' and nothing else so that it is easy for me to parse.\nINPUT: "
        + input_text
    )

if __name__ == "__main__":
    args = parse_args()
    algorithms = args.algorithms.split(",")

    print(f"PID: {os.getpid()}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Stage 1: Paraphrasing
    print("=== Stage 1: Generate reference text ===")
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        do_sample=False
    )
    
    for algorithm in algorithms:
        input_file = os.path.join(args.input_dir, f'{algorithm}_response.json')
        ref_output_file = os.path.join(args.result_dir, f'{algorithm}_ref.json')

        with open(input_file, 'r') as f:
            lines = f.readlines()

        last_line = 0
        if os.path.exists(ref_output_file):
            with open(ref_output_file, 'r') as f:
                last_line = sum(1 for _ in f)

        with open(ref_output_file, 'a') as out_f:
            for i, line in enumerate(tqdm(lines[last_line:], desc=f"Stage 1: {algorithm}", unit="line")):
                item = json.loads(line)
                prompt = item['prompt']
                watermarked_text = item['watermarked_text']
                unwatermarked_text = item['unwatermarked_text']

                input_text = fill_parapharseprompt(watermarked_text)
                messages = [
                    {"role": "system", "content": "You are a helpful rewriter."},
                    {"role": "user", "content": input_text},
                ]
                outputs = pipeline(messages, max_new_tokens=256, do_sample=False)
                output_text = outputs[0]["generated_text"][-1]["content"]

                response_item = {
                    'prompt': prompt,
                    'watermarked_text': watermarked_text,
                    'unwatermarked_text': unwatermarked_text,
                    'ref_text': output_text,
                }
                out_f.write(json.dumps(response_item) + '\n')

    del pipeline
    torch.cuda.empty_cache()

    # Stage 2: Self-Information Blanking
    print("=== Stage 2: Self-Information Blanking ===")

    class SelfInformationCalculator:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self._prepare_model()
        
        def _prepare_model(self):
            self.model.eval()
            print('Model and tokenizer loaded successfully.')

        def calculate_self_information(self, text: str):
            with torch.no_grad():
                encoding = self.tokenizer(text, add_special_tokens=False, return_tensors='pt').to(self.model.device)
                outputs = self.model(**encoding)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                self_info = -torch.log(probs)

            input_ids = encoding['input_ids']
            input_ids_expaned = input_ids[:, 1:].unsqueeze(-1)

            tokens = [self.tokenizer.decode(token_) for token_ in input_ids.squeeze().tolist()[1:]]
            self_info_values = self_info[:, :-1].gather(-1, input_ids_expaned).squeeze(-1).squeeze(0).tolist()

            return tokens, self_info_values

        def transform_tokens(self, tokens, self_info_values, threshold_low):
            percentile = np.percentile(self_info_values, threshold_low)
            transformed_tokens = []
            temp_tokens = []

            for token, self_info in zip(tokens, self_info_values):
                if self_info > percentile:
                    if temp_tokens:
                        transformed_tokens.append(f"({' '.join(temp_tokens)})")
                        temp_tokens = []
                    transformed_tokens.append('_')
                else:
                    if temp_tokens:
                        transformed_tokens.append(f"({' '.join(temp_tokens)})")
                        temp_tokens = []
                    transformed_tokens.append(token)

            if temp_tokens:
                transformed_tokens.append(f"({' '.join(temp_tokens)})")

            return transformed_tokens

    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    calculator = SelfInformationCalculator(model=model, tokenizer=tokenizer)
    threshold = args.threshold

    for algorithm in algorithms:
        ref_output_file = os.path.join(args.result_dir, f'{algorithm}_ref.json')
        blank_output_file = os.path.join(args.result_dir, f'{algorithm}_blank.json')

        processed_lines = 0
        if os.path.exists(blank_output_file):
            with open(blank_output_file, 'r') as f:
                processed_lines = sum(1 for _ in f)
            print(f"Found {processed_lines} processed lines for {algorithm}")

        with open(ref_output_file, 'r') as f:
            lines = f.readlines()

        with open(blank_output_file, 'a') as out_f:
            for line in tqdm(lines[processed_lines:], desc=f"Stage 2: {algorithm}", unit="line"):
                item = json.loads(line)
                prompt = item['prompt']
                watermarked_text = item['watermarked_text']
                unwatermarked_text = item['unwatermarked_text']
                ref_text = item['ref_text']

                tokens, self_info_values = calculator.calculate_self_information(watermarked_text)
                transformed_tokens = calculator.transform_tokens(tokens, self_info_values, threshold)
                blank_text = "".join(transformed_tokens)

                response_item = {
                    'prompt': prompt,
                    'watermarked_text': watermarked_text,
                    'unwatermarked_text': unwatermarked_text,
                    'ref_text': ref_text,
                    'blank_text': blank_text
                }
                out_f.write(json.dumps(response_item) + '\n')

    del model, tokenizer, calculator
    torch.cuda.empty_cache()
