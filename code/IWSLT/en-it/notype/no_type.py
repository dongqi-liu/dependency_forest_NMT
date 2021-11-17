from supar import Parser
import torch
import numpy as np
import random
from torch.nn import functional as F

def dependency_tree_without_type(parser,src_batch_list,src_sample_list):
    final_dataset = []
    max_len = 0
    dataset = parser.predict(src_sample_list, lang='en', prob=False,forest=True,verbose=False)
    for forest in dataset.forest:
        max_len = max(max_len,forest.shape[0])
    for src, forest in zip(src_batch_list,dataset.forest):
        relationship = F.pad(forest.softmax(-1).max(-1)[0].squeeze(-1),(0, max_len - len(src) + 2, 0, max_len - len(src) + 2), "constant", 0)
        total_relationship = torch.stack([relationship,relationship,relationship,relationship,relationship,relationship,relationship,
                                              relationship,relationship,relationship,relationship,relationship,relationship,relationship,relationship,
                                              relationship],dim=0)
        total_relationship = torch.nan_to_num(total_relationship)
        final_dataset.append(total_relationship)
    return final_dataset
