import numpy as np


dataset_name = "reddit"
x_text = np.load(f'/root/autodl-tmp/lht/GraphAdapter/{dataset_name}/x_text.npy')
y = np.load(f'/root/autodl-tmp/lht/GraphAdapter/{dataset_name}/y.npy')
print(x_text[2])
print(y[0])