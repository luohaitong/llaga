import torch

autodl_prefix = "/root/autodl-tmp/lht/LLaGA/"
data = torch.load(autodl_prefix + "cora/processed_data.pt")
labels=data.label_texts
short_labels = [l.split('_')[0] for l in labels]
print(labels)
print(short_labels)