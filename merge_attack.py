import os
import json
import torch
import numpy as np
from tqdm import tqdm
import argparse
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run paraphrasing, self-information blanking, and attack pipeline.")
    # Directories
    parser.add_argument('--orig_input_dir', type=str, required=True, default='/root/Self-information-Rewrite-Attack-main/dataset/c4/watermarked/',
                        help='Directory containing initial {algorithm}_response.json input files for Stage 1.')
    parser.add_argument('--intermediate_dir', type=str, required=True, default='/root/Self-information-Rewrite-Attack-main/dataset/c4/watermarked_processed/',
                        help='Directory to store intermediate _ref.json and _blank.json files, and for input to Stage 3.')
    parser.add_argument('--final_output_dir', type=str, required=True, default='/root/Self-information-Rewrite-Attack-main/dataset/c4/attack_results/',
                        help='Directory to store final _attack.json output files from Stage 3.')

    # Model paths
    parser.add_argument('--paraphrase_model_path', type=str, required=True, default='/root/models/meta-llama/Llama-3.2-3B-Instruct/',
                        help='Path to the HuggingFace model for Stage 1 (Paraphrasing) and Stage 2 (Self-Information Blanking).')
    parser.add_argument('--attack_model_path', type=str, required=True, default='/root/models/meta-llama/Llama-3.2-3B-Instruct/',
                        help='Path to the HuggingFace model for Stage 3 (Attack).')

    # Parameters
    parser.add_argument('--threshold', type=int, default=30, help='Percentile threshold for self-information blanking in Stage 2.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU IDs to use, e.g., "0,1,2".')
    parser.add_argument('--algorithms', type=str, default='KGW', # Example: 'KGW,Unigram,UPV'
                        help='Comma-separated list of watermark algorithms to process.')
    return parser.parse_args()

def fill_parapharse_prompt(input_text):
    return (
        "You are a paraphraser. You are given an input passage 'INPUT'. "
        "You should paraphrase 'INPUT' to print 'OUTPUT'. 'OUTPUT' should be diverse and different "
        "as much as possible from 'INPUT' and should not copy any part verbatim from 'INPUT'. "
        "However, 'OUTPUT' should preserve the information in the INPUT. "
        "You should print 'OUTPUT' and nothing else so that it is easy for me to parse.\nINPUT: "
        + input_text
    )

def fill_attack_prompt(reference_text, blank_text):
    prompt = (
        "You will be shown one reference paragraph and one incomplete paragraph.\n"
        "Your task is to write a complete paragraph using incomplete paragraph.\n"
        "The complete paragraph should have similar length with reference paragraph.\n"
        "You need to include all the information in the reference. \n"
        "But do not take the expression and words in the reference paragraph.\n"
        "You should only answer the complete paragraph.\n"
        f"reference: {reference_text}\n"
        f"incomplete pragraph: {blank_text}\n" # "paragraph" was misspelled in original
    )
    return prompt

class SelfInformationCalculator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._prepare_model()

    def _prepare_model(self):
        self.model.eval()
        print('SelfInformationCalculator: Model and tokenizer loaded successfully.')

    def calculate_self_information(self, text: str):
        with torch.no_grad():
            if not text.strip():
                return [], []
            encoding = self.tokenizer(text, add_special_tokens=False, return_tensors='pt').to(self.model.device)
            if encoding['input_ids'].shape[1] == 0:
                return [],[]
            outputs = self.model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            self_info = -torch.log(probs) 

        input_ids = encoding['input_ids']
        if input_ids.shape[1] <= 1:
             return [], [] 

        input_ids_expanded = input_ids[:, 1:].unsqueeze(-1) 

        tokens = [self.tokenizer.decode(token_) for token_ in input_ids.squeeze().tolist()[1:]]
        self_info_values = self_info[:, :-1].gather(-1, input_ids_expanded).squeeze(-1).squeeze(0).tolist()

        return tokens, self_info_values

    def transform_tokens(self, tokens, self_info_values, threshold_low):
        if not tokens or not self_info_values: 
            return []
        percentile = np.percentile(self_info_values, threshold_low)
        transformed_tokens = []
        current_low_si_chunk = []

        for token, self_info in zip(tokens, self_info_values):
            if self_info <= percentile: 
                current_low_si_chunk.append(token)
            else:
                if current_low_si_chunk: 
                    transformed_tokens.append(f"({' '.join(current_low_si_chunk)})")
                    current_low_si_chunk = []
                transformed_tokens.append('_') 

        if current_low_si_chunk: 
            transformed_tokens.append(f"({' '.join(current_low_si_chunk)})")

        return transformed_tokens


def main():
    args = parse_args()
    algorithms = args.algorithms.split(",")

    print(f"Master PID: {os.getpid()}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    num_gpus_visible = len(args.gpu.split(',')) if args.gpu else 0
    print(f"CUDA_VISIBLE_DEVICES set to: {args.gpu}. Number of GPUs made visible: {num_gpus_visible}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available, will use CPU if model mapping allows.")
    elif num_gpus_visible == 0 and args.gpu.strip() != "": # User specified GPUs but none are valid/found
        print(f"WARNING: CUDA_VISIBLE_DEVICES is set to '{args.gpu}' but PyTorch reports CUDA not available or no GPUs found after setting.")


    os.makedirs(args.intermediate_dir, exist_ok=True)
    os.makedirs(args.final_output_dir, exist_ok=True)

    # === Stage 1: Generate reference text (Paraphrasing) ===
    print("\n=== Stage 1: Generate reference text (Paraphrasing) ===")
    # Check if paraphrase_model_path is valid before loading
    if not os.path.exists(args.paraphrase_model_path):
        print(f"[ERROR] Paraphrase model path not found: {args.paraphrase_model_path}")
        return

    paraphrase_pipeline = transformers.pipeline(
        "text-generation",
        model=args.paraphrase_model_path,
        model_kwargs={"torch_dtype": torch.bfloat16}, 
        device_map="auto", 
        do_sample=False 
    )

    for algorithm in algorithms:
        input_file_s1 = os.path.join(args.orig_input_dir, f'{algorithm}_response.json')
        ref_output_file = os.path.join(args.intermediate_dir, f'{algorithm}_ref.json')

        if not os.path.exists(input_file_s1):
            print(f"[S1 Warning] Skipping {algorithm}: Input file not found → {input_file_s1}")
            continue

        with open(input_file_s1, 'r') as f:
            lines_s1 = f.readlines()

        last_line_s1 = 0
        if os.path.exists(ref_output_file):
            try:
                with open(ref_output_file, 'r') as f_check:
                    last_line_s1 = sum(1 for _ in f_check)
                print(f"[S1 Info] Resuming {algorithm} from line {last_line_s1} for reference text generation.")
            except Exception as e:
                print(f"[S1 Warning] Could not read existing ref file {ref_output_file}, starting from scratch. Error: {e}")
                last_line_s1 = 0


        with open(ref_output_file, 'a') as out_f_s1:
            for i, line in enumerate(tqdm(lines_s1[last_line_s1:], desc=f"Stage 1 [{algorithm}]", unit="line")):
                try:
                    item = json.loads(line)
                    prompt_text = item['prompt']
                    watermarked_text = item['watermarked_text']
                    unwatermarked_text = item.get('unwatermarked_text', "") # Handle missing key

                    paraphrase_input_text = fill_parapharse_prompt(watermarked_text)
                    messages = [
                        {"role": "system", "content": "You are a helpful rewriter."},
                        {"role": "user", "content": paraphrase_input_text},
                    ]
                    outputs = paraphrase_pipeline(messages, max_new_tokens=256, do_sample=False)
                    generated_content = ""
                    if outputs and outputs[0]["generated_text"]:
                        if isinstance(outputs[0]["generated_text"], list) and len(outputs[0]["generated_text"]) > 0:
                            if isinstance(outputs[0]["generated_text"][-1], dict) and "content" in outputs[0]["generated_text"][-1]:
                                generated_content = outputs[0]["generated_text"][-1]["content"]
                            else: 
                                generated_content = str(outputs[0]["generated_text"][-1])
                        elif isinstance(outputs[0]["generated_text"], str): # if pipeline returns full string
                             print(f"[S1 Warning] Unexpected output format from paraphrase_pipeline for item {i}. Full output: {outputs[0]['generated_text']}")
                             generated_content = str(outputs[0]["generated_text"]) # Fallback

                    response_item = {
                        'prompt': prompt_text,
                        'watermarked_text': watermarked_text,
                        'unwatermarked_text': unwatermarked_text,
                        'ref_text': generated_content,
                    }
                    out_f_s1.write(json.dumps(response_item) + '\n')
                except json.JSONDecodeError:
                    print(f"[S1 Error] Could not parse JSON from line {i+last_line_s1} in {input_file_s1}. Skipping.")
                except Exception as e:
                    print(f"[S1 Error] Error processing line {i+last_line_s1} for {algorithm}: {e}")
    
    del paraphrase_pipeline
    torch.cuda.empty_cache()
    print("=== Stage 1: Completed ===")

    # === Stage 2: Self-Information Blanking ===
    print("\n=== Stage 2: Self-Information Blanking ===")
    if not os.path.exists(args.paraphrase_model_path): # Re-check as it could be same model
        print(f"[ERROR] Blanking model path not found: {args.paraphrase_model_path}")
        return

    blanking_model = AutoModelForCausalLM.from_pretrained(args.paraphrase_model_path, device_map="auto", torch_dtype=torch.bfloat16)
    blanking_tokenizer = AutoTokenizer.from_pretrained(args.paraphrase_model_path)
    calculator = SelfInformationCalculator(model=blanking_model, tokenizer=blanking_tokenizer)
    threshold = args.threshold

    for algorithm in algorithms:
        ref_output_file_s2 = os.path.join(args.intermediate_dir, f'{algorithm}_ref.json')
        blank_output_file = os.path.join(args.intermediate_dir, f'{algorithm}_blank.json')

        if not os.path.exists(ref_output_file_s2):
            print(f"[S2 Warning] Skipping {algorithm}: Ref file not found for blanking → {ref_output_file_s2}")
            continue

        processed_lines_s2 = 0
        if os.path.exists(blank_output_file):
            try:
                with open(blank_output_file, 'r') as f_check:
                    processed_lines_s2 = sum(1 for _ in f_check)
                print(f"[S2 Info] Resuming {algorithm} from line {processed_lines_s2} for blanking.")
            except Exception as e:
                print(f"[S2 Warning] Could not read existing blank file {blank_output_file}, starting from scratch. Error: {e}")
                processed_lines_s2 = 0


        with open(ref_output_file_s2, 'r') as f_s2:
            lines_s2 = f_s2.readlines()

        with open(blank_output_file, 'a') as out_f_s2:
            for line in tqdm(lines_s2[processed_lines_s2:], desc=f"Stage 2 [{algorithm}]", unit="line"):
                try:
                    item = json.loads(line)
     
                    watermarked_text = item['watermarked_text'] 

                    tokens, self_info_values = calculator.calculate_self_information(watermarked_text)
                    if not tokens: 
                         blank_text = "" 
                    else:
                        transformed_tokens = calculator.transform_tokens(tokens, self_info_values, threshold)
                        blank_text = "".join(transformed_tokens) 


                    response_item = item.copy()
                    response_item['blank_text'] = blank_text
                    out_f_s2.write(json.dumps(response_item) + '\n')
                except json.JSONDecodeError:
                    print(f"[S2 Error] Could not parse JSON from line in {ref_output_file_s2} during Stage 2 for {algorithm}. Skipping.")
                except Exception as e:
                    print(f"[S2 Error] Error processing line for {algorithm} during blanking: {e}")

    del blanking_model, blanking_tokenizer, calculator
    torch.cuda.empty_cache()
    print("=== Stage 2: Completed ===")

    # === Stage 3: Attack ===
    print("\n=== Stage 3: Attack using blanked text ===")
    if not os.path.exists(args.attack_model_path):
        print(f"[ERROR] Attack model path not found: {args.attack_model_path}")
        return

    attack_pipeline = transformers.pipeline(
        "text-generation",
        model=args.attack_model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        do_sample=False 
    )

    for algorithm in algorithms:
        input_file_s3 = os.path.join(args.intermediate_dir, f'{algorithm}_blank.json')
        output_file_s3 = os.path.join(args.final_output_dir, f'{algorithm}_attack.json')

        if not os.path.exists(input_file_s3):
            print(f"[S3 Warning] Skipping {algorithm}: Blank input file not found → {input_file_s3}")
            continue

        with open(input_file_s3, 'r') as f_s3:
            lines_s3 = f_s3.readlines()

        last_line_s3 = 0
        if os.path.exists(output_file_s3):
            try:
                with open(output_file_s3, 'r') as f_check:
                    last_line_s3 = sum(1 for _ in f_check)
                print(f"[S3 Info] Resuming {algorithm} from line {last_line_s3} for attack generation.")
            except Exception as e:
                 print(f"[S3 Warning] Could not read existing attack file {output_file_s3}, starting from scratch. Error: {e}")
                 last_line_s3 = 0

        with open(output_file_s3, 'a') as out_f_s3: 
            for line in tqdm(lines_s3[last_line_s3:], desc=f"Stage 3 [{algorithm}]", unit="line"):
                try:
                    item = json.loads(line)
                    blank_text = item['blank_text']
                    ref_text = item['ref_text'] 

                    attack_input_text = fill_attack_prompt(ref_text, blank_text)
                    messages = [{"role": "user", "content": attack_input_text}]
            
                    outputs = attack_pipeline(messages, max_new_tokens=256, do_sample=False)
                    
                    generated_content_attack = ""
                    if outputs and outputs[0]["generated_text"]:
                         if isinstance(outputs[0]["generated_text"], list) and len(outputs[0]["generated_text"]) > 0:
                            if isinstance(outputs[0]["generated_text"][-1], dict) and "content" in outputs[0]["generated_text"][-1]:
                                generated_content_attack = outputs[0]["generated_text"][-1]["content"]
                            else:
                                generated_content_attack = str(outputs[0]["generated_text"][-1])
                         elif isinstance(outputs[0]["generated_text"], str):
                            print(f"[S3 Warning] Unexpected output format from attack_pipeline. Full output: {outputs[0]['generated_text']}")
                            generated_content_attack = str(outputs[0]["generated_text"]) # Fallback


                    response_item = item.copy()
                    response_item['attack_text'] = generated_content_attack
                    out_f_s3.write(json.dumps(response_item) + '\n')
                except json.JSONDecodeError:
                    print(f"[S3 Error] Could not parse JSON from line in {input_file_s3} during Stage 3 for {algorithm}. Skipping.")
                except Exception as e:
                    print(f"[S3 Error] Error processing line for {algorithm} during attack: {e}")


    del attack_pipeline
    torch.cuda.empty_cache()
    print("=== Stage 3: Completed ===")
    print("All stages finished.")

if __name__ == "__main__":
    main()