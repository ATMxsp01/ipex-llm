import os
from transformers import AutoTokenizer
from ipex_llm.transformers import AutoModelForCausalLM
from datasets import load_dataset
import json
import argparse

result_path = 'prompt_count/'
count_path = f'{result_path}count.json'
maxlen = 4096
model_name = 'mistral-7B-instruct-v0.2'
model_path = '/home/arda/xs/models/Mistral-7B-Instruct-v0.2'
# model_name = 'Llama-2-13b-chat-hf'
# model_path = '/home/arda/llm_models/Llama-2-13b-chat-hf'

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='qasper', help="Dataset to evaluate on")
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    datas = load_dataset('THUDM/LongBench', f'{args.dataset}_e', split='test')
    


    #model = AutoModelForCausalLM.from_pretrained(
    #    model_path,
    #    load_in_4bit=True,
    #    #low_cpu_mem_usage=True,
    #    device_map="auto",
    #    use_cache=True,
    #    #use_flash_attention_2=True
    #)
    #model = model.to('xpu')
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        use_fast=False,
    )

    dataset_raw = {}
    dataset_info = {} 
    prompt_cnt = {}
    if os.path.exists(count_path):
        with open(count_path, "r", encoding = "utf-8") as f:
            prompt_cnt = json.load(f)

    cnt_all = 0
    cnt_input = 0
    cnt_context = 0
    for i, data in enumerate(datas):
        tokenized_prompt = tokenizer(data['input'], truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(data['input'], truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        len_input = len(tokenized_prompt)
        tokenized_prompt = tokenizer(data['context'], truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(data['context'], truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        len_context = len(tokenized_prompt)
        len_all = len_input + len_context
        
        dataset_info[i] = {
            'input' : len_input, 
            'context' : len_context, 
            'all' : len_all
        }

        dataset_raw[i] = {
            'input' : data['input'],
            'context' : data['context']
        }

        if len_input <= maxlen:
            cnt_input += 1
        if len_context <= maxlen:
            cnt_context += 1
        if len_all <= maxlen:
            cnt_all += 1
    prompt_cnt[args.dataset] = {
        'all' : cnt_all,
        'input' : cnt_input,
        'context' : cnt_context
    }

    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    dataset_info_path = f'{result_path}{args.dataset}.json'
    with open(dataset_info_path, "w", encoding = "utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=4)
    with open(count_path, "w", encoding = "utf-8") as f:
        json.dump(prompt_cnt, f, ensure_ascii=False, indent=4)
    #raw_path = f'{result_path}{args.dataset}_raw.json'
    #with open(raw_path, "w", encoding = "utf-8") as f:
    #    json.dump(dataset_raw, f, ensure_ascii=False, indent=4)