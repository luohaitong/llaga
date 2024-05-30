import torch
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
from async_api import process_api_requests_from_file
import re

async def call_async_api(request_filepath, save_filepath, request_url, api_key, max_request_per_minute, max_tokens_per_minute, sp, ss):
    await process_api_requests_from_file(
            requests_filepath=request_filepath,
            save_filepath=save_filepath,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(max_request_per_minute),
            max_tokens_per_minute=float(max_tokens_per_minute),
            token_encoding_name='cl100k_base',
            max_attempts=int(2),
            logging_level=int(logging.INFO),
            seconds_to_pause=sp,
            seconds_to_sleep=ss
        )
    
def efficient_openai_text_api(filename, savepath, sp, ss, api_key="sk-DuWNMZghRV110XRAEe2dF1A5B3264b4a9704547fEbB03c32", rewrite = True):
    #if not osp.exists(savepath) or rewrite:
    asyncio.run(
        call_async_api(
            filename, save_filepath=savepath,
            #request_url="https://api.openai.com/v1/chat/completions",
            request_url="https://openkey.cloud/v1/chat/completions",
            #request_url="https://key.wenwen-ai.com/v1/chat/completions",
            api_key=api_key,
            max_request_per_minute=1000,
            max_tokens_per_minute=90000,
            sp=sp,
            ss=ss
        )
    )

def generate_prompt(generate_nums, filename, data_path):
    data = torch.load(data_path)
    titles = data.title
    abstracts = data.abs
    prompts = []
    num = str(generate_nums)
    print("dataset number:", len(titles))
    for i in range(len(titles)):
        title = titles[i]
        abstract = abstracts[i]
        prompt = f"Give you a title and a description of a paper, your task is to generate new versions of the title and description while maintaining the original meaning. Modify the text as much as possible, including grammar, wording, word order, and sentence structure. The output should be returned only in JSON format for each version, with the keys 'title' and 'description'. Please return to me {num} different versions directly without other texts.\n The title is: {title}. \n The abstract is: {abstract}."
        prompts.append(prompt)
    
    jobs = generate_chat_input_file(prompts)
    with open(filename, "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")



def generate_chat_input_file(input_text, model_name = 'gpt-3.5-turbo'):
    jobs = []
    for i, text in enumerate(input_text):
        obj = {}
        obj['model'] = model_name
        obj['messages'] = [
            {
                'role': 'user',
                'content': text 
            }
        ]
        jobs.append(obj)
    return jobs 

def process_openai_out(openai_out_file, syn_out_path):

    openai_result = []
    with open(openai_out_file, 'r') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            idx = json_obj[-1]
            response = json_obj[1]['choices'][0]['message']['content']
            openai_result.append((response, idx))

    openai_result = sorted(openai_result, key=lambda x: x[-1])
    synthesized_titles = []
    synthesized_descriptions = []
    for result in openai_result:
        result = str(result)
        titles = re.findall('"title":\s*"([^"]+)"', result)
        descriptions = re.findall('"description":\s*"([^"]+)"', result)
        synthesized_titles.append(titles)
        synthesized_descriptions.append(descriptions)
    save_dict = {'syn_title': synthesized_titles, 'syn_abs': synthesized_descriptions}
    torch.save(save_dict, syn_out_path)

if __name__ == '__main__':
    generate_nums = 3
    dataset_name = 'cora'
    data_path = f'../dataset/{dataset_name}/processed_data.pt'
    syn_data_path = f'../dataset/{dataset_name}/openai_syn_data.pt'

    input_file_name = f"../openai_out/{dataset_name}_text_augment_input.json"
    output_file_name = f"../openai_out/{dataset_name}_text_augment_output.json"
    #api_key = "sk-DuWNMZghRV110XRAEe2dF1A5B3264b4a9704547fEbB03c32"
    api_key = "sk-SrJ4zosDYdStagkXF36444F8Dd104eD7A625922aAfB86a44"
    prompt_exist = True

    if not prompt_exist:
        generate_prompt(generate_nums = generate_nums, filename=input_file_name, data_path = data_path)
    #efficient_openai_text_api(input_file_name, output_file_name, sp=60, ss=1.5, api_key= api_key, rewrite=False)
    process_openai_out(output_file_name, syn_data_path)

