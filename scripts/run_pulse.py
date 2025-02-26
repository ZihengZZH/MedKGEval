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
    'pulse': '../llm-weights/PULSE-7bv5',
}


def load_dataset(infile):
    with open(infile, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    return data_dict['qa-list'], data_dict['prompt']


def main(args):

    if args.model not in MODEL_MAP:
        raise ValueError("Model not found in MODEL_MAP")

    model_name_or_path = MODEL_MAP[args.model]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, low_cpu_mem_usage=True, device_map='auto',
        torch_dtype=torch.float16, trust_remote_code=True
    )
    model.eval()

    data_list, prompt = load_dataset(args.input)
    data_output = list()

    first_instruction = "Instructions: You are Helper, a large language model full of intelligence. Respond conversationally."

    for idx, data in enumerate(data_list):
        start = time.time()
        question = prompt + data['question']

        input_ids = tokenizer(
            first_instruction,
            add_special_tokens=False
        ).input_ids + [tokenizer.convert_tokens_to_ids("</s>")]

        input_ids += tokenizer("User: " + question).input_ids
        input_ids += [tokenizer.convert_tokens_to_ids("</s>")]

        model_inputs = tokenizer.pad(
            {"input_ids": [input_ids + tokenizer("Helper: ").input_ids[:1]]},
            return_tensors="pt",
        )

        inputs = model_inputs.input_ids[:, -1024:]
        attention_mask = model_inputs.attention_mask[:, -1024:]

        max_length = inputs.shape[1] + 512
        min_length = inputs.shape[1] + 1  # add eos

        outputs = model.generate(
            inputs=inputs.cuda(),
            attention_mask=attention_mask.cuda(),
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            top_k=6,
            top_p=0.1,
            temperature=0.7,
            num_return_sequences=1,
            eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
        )

        outputs_token = outputs[0].tolist()
        outputs_token = outputs_token[inputs.shape[1]:]

        response = tokenizer.decode(
            outputs_token,
            skip_special_tokens=False
        )

        response = response.strip()
        if response[:3] == "<s>":
            response = response[3:]
        if response[-4:] == "</s>":
            response = response[:-4]
        if response.startswith(': '):
            response = response[2:]
        response = response.strip()

        time_cost = time.time() - start
        num_gen_tokens = len(outputs_token)

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
