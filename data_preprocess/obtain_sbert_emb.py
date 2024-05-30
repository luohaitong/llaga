import sys
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def sbert(model_type, device):
    model = SentenceTransformer(model_type, device=device)
    return model

def get_sbert_embedding(model_type, texts, device):
    if model_type == 'sbert':
        model_type = 'all-MiniLM-L6-v2'
    sbert_model = sbert(model_type, f'cuda:{device}')
    sbert_embeds = sbert_model.encode(texts, batch_size=8, show_progress_bar=True)
    return torch.tensor(sbert_embeds)

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    print('Dataset: {}'.format(dataset_name))

    data_path = f'../dataset/{dataset_name}/openai_syn_data.pt'
    data = torch.load(data_path)
    title = data['syn_title']
    abs = data['syn_abs']
    print(len(title))
    print(len(abs))
    concat_title = []
    for i in range(len(title)):
        concat_title.append(title[i]+abs[i])
    output_embedding = get_sbert_embedding("sbert", concat_title, 0)
    torch.save(output_embedding, 'syn_output_pretrained_embedding.pt')




