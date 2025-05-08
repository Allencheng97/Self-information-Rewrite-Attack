import os
import json
import torch
import numpy as np
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from visualize.font_settings import FontSettings
from visualize.visualizer import ContinuousVisualizer
from visualize.legend_settings import ContinuousLegendSettings
from visualize.page_layout_settings import PageLayoutSettings
from visualize.color_scheme import ColorSchemeForContinuousVisualization
print(f"PID: {os.getpid()}")
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def fill_parapharseprompt(input_text):
    return (
        "You are a paraphraser. You are given an input passage 'INPUT'. "
        "You should paraphrase 'INPUT' to print 'OUTPUT'. 'OUTPUT' should be diverse and different "
        "as much as possible from 'INPUT' and should not copy any part verbatim from 'INPUT'. "
        "However, 'OUTPUT' should preserve the information in the INPUT. "
        "You should print 'OUTPUT' and nothing else so that it is easy for me to parse.\nINPUT: "
        + input_text
    )

# Stage 1: paraphrasing
print("=== Stage 1: Paraphrasing ===")
pipeline = transformers.pipeline(
    "text-generation",
    model="/mnt/datadisk/syj/models/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    do_sample=False
)

algorithms = ['KGW','Unigram','UPV','DIP','EWD','EXP','SIR']
for algorithm in algorithms:
    input_file = f'/mnt/datadisk/syj/MarkLLM/c4_adaptive/adaptive_watermark_response_500.json'
    output_file = f'/mnt/datadisk/syj/MarkLLM/c4_adaptive/adaptive_watermark_response_500_ref_llama3_8b.json.json'

    with open(input_file, 'r') as f:
        lines = f.readlines()

    last_line = 0
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            last_line = sum(1 for _ in f)

    with open(output_file, 'a') as out_f:
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

# Clean GPU memory before Stage 2
del pipeline
torch.cuda.empty_cache()

# Stage 2: self-information blanking
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

model = AutoModelForCausalLM.from_pretrained(
    "/mnt/datadisk/syj/models/Meta-Llama-3-8B-Instruct",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/mnt/datadisk/syj/models/Meta-Llama-3-8B-Instruct")
calculator = SelfInformationCalculator(model=model, tokenizer=tokenizer)

for algorithm in algorithms:
    input_file = f'/mnt/datadisk/syj/MarkLLM/c4_adaptive/adaptive_watermark_response_500_ref_llama3_8b.json'
    output_file = f'/mnt/datadisk/syj/MarkLLM/c4_adaptive/adaptive_watermark_response_500_blank_llama3_8b.json'

    processed_lines = 0
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            processed_lines = sum(1 for _ in f)
        print(f"Found {processed_lines} processed lines for {algorithm}")

    with open(input_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'a') as out_f:
        for line in tqdm(lines[processed_lines:], desc=f"Stage 2: {algorithm}", unit="line"):
            item = json.loads(line)
            prompt = item['prompt']
            watermarked_text = item['watermarked_text']
            unwatermarked_text = item['unwatermarked_text']
            ref_text = item['ref_text']

            tokens, self_info_values = calculator.calculate_self_information(watermarked_text)
            transformed_tokens = calculator.transform_tokens(tokens, self_info_values, 30)
            blank_text = "".join(transformed_tokens)

            response_item = {
                'prompt': prompt,
                'watermarked_text': watermarked_text,
                'unwatermarked_text': unwatermarked_text,
                'ref_text': ref_text,
                'blank_text': blank_text
            }
            out_f.write(json.dumps(response_item) + '\n')

# Clean up VRAM after stage 2
del model, tokenizer, calculator
torch.cuda.empty_cache()
