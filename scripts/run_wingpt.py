import json
import time
import numpy as np
import argparse
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_MAP = {
    'wingpt2': '../llm-weights/WiNGPT2-Llama-3-8B-Chat',
}


def load_dataset(infile):
    with open(infile, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    return data_dict['qa-list'], data_dict['prompt']


def main(args):

    if args.model not in MODEL_MAP:
        raise ValueError("Model not found in MODEL_MAP")

    model_name_or_path = MODEL_MAP[args.model]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, low_cpu_mem_usage=True, device_map='auto'
    )
    model = model.eval()

    data_list, prompt = load_dataset(args.input)
    data_output = list()

    for idx, data in enumerate(data_list):
        start = time.time()
        question = prompt + data['question']

        text = 'User：%s<|end_of_text|>\n Assistant：' % question
        inputs = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
        outputs = model.generate(
            inputs=inputs,
            max_new_tokens=64,
            repetition_penalty=1.1
        )
        response = tokenizer.decode(outputs[0])
        response = response[len(text):].replace('<|end_of_text|>', '').strip()

        time_cost = time.time() - start
        num_gen_tokens = outputs[0].shape[0] - inputs[0].shape[0]

        print('Prompt %d/%d' % (idx+1, len(data_list)))
        print(question)
        print(">", response)
        print("\n==================================\n")

        temp_dict = data.copy()
        temp_dict['generation'] = response
        temp_dict['tokens'] = num_gen_tokens
        temp_dict['timecost'] = '%.4f' % time_cost
        data_output.append(temp_dict)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data_output, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    main(args)
