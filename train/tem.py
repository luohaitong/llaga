import torch
autodl_prefix = "/root/autodl-tmp/lht/LLaGA/"
data_path = autodl_prefix + "pubmed/processed_data.pt"
data = torch.load(data_path)
title = data.title[0]
label = data.label_texts[0]
print(title)
print(label)

autodl_prefix = "/root/autodl-tmp/lht/LLaGA/"
data_path = autodl_prefix + "cora/processed_data.pt"
data = torch.load(data_path)
title = data.title[0]
label = data.label_texts[0]
print(title)
print(label)
