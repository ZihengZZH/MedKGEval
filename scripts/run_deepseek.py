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
    'ds-1.5b': '../llm-weights/DeepSeek-R1-Distill-Qwen-1.5B',
    'ds-7b': '../llm-weights/DeepSeek-R1-Distill-Qwen-7B',
    'ds-14b': '../llm-weights/DeepSeek-R1-Distill-Qwen-14B',
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
    model.eval()

    data_list, prompt = load_dataset(args.input)
    data_output = list()

    for idx, data in enumerate(data_list):
        start = time.time()
        question = prompt + data['question']
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': question}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512 * 2
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        time_cost = time.time() - start
        num_gen_tokens = generated_ids[0].shape[0]

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
