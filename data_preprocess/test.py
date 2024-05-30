import json
import re

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
        print("title:", titles)
        print("descriptions:", descriptions)
        print("--------")
    save_dict = {'syn_title': synthesized_titles, 'syn_abs': synthesized_descriptions}


dataset_name = 'cora'
syn_data_path = f'../dataset/{dataset_name}/openai_syn_data.pt'
output_file_name = f"../openai_out/{dataset_name}_text_augment_output.json"

process_openai_out(output_file_name, syn_data_path)