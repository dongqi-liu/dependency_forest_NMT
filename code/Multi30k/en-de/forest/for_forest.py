from supar import Parser
import torch
import numpy as np
import random
from torch.nn import functional as F

def dependency_forest(parser,src_batch_list,src_sample_list):
    final_dataset = []
    max_len = 0
    dataset = parser.predict(src_sample_list, lang='en', prob=False,forest=True,verbose=False)
    for forest in dataset.forest:
        max_len = max(max_len,forest.shape[0])
    for src, forest in zip(src_batch_list,dataset.forest):
        relationship_1 = F.pad(forest.softmax(-1).index_select(dim=-1,index=torch.tensor([43])).sum(-1).squeeze(-1),(0, max_len - len(src) + 2, 0, max_len - len(src) + 2), "constant", 0)
        relationship_2 = F.pad(forest.softmax(-1).index_select(dim=-1,index = torch.tensor([6,7,11])).sum(-1).squeeze(-1),(0, max_len - len(src) + 2, 0, max_len - len(src) + 2), "constant", 0)
        relationship_3 = F.pad(forest.softmax(-1).index_select(dim=-1,index = torch.tensor([9,45,1,32])).sum(-1).squeeze(-1),(0, max_len - len(src) + 2, 0, max_len - len(src) + 2), "constant", 0)
        relationship_4 = F.pad(forest.softmax(-1).index_select(dim=-1,index = torch.tensor([17,20,33])).sum(-1).squeeze(-1),(0, max_len - len(src) + 2, 0, max_len - len(src) + 2), "constant", 0)
        relationship_5 = F.pad(forest.softmax(-1).index_select(dim=-1,index = torch.tensor([12,13])).sum(-1).squeeze(-1),(0, max_len - len(src) + 2, 0, max_len - len(src) + 2), "constant", 0)
        relationship_6 = F.pad(forest.softmax(-1).index_select(dim=-1,index = torch.tensor([26,27])).sum(-1).squeeze(-1),(0, max_len - len(src) + 2, 0, max_len - len(src) + 2), "constant", 0)
        relationship_7 = F.pad(forest.softmax(-1).index_select(dim=-1,index = torch.tensor([8])).sum(-1).squeeze(-1),(0, max_len - len(src) + 2, 0, max_len - len(src) + 2), "constant", 0)
        relationship_8 = F.pad(forest.softmax(-1).index_select(dim=-1,index = torch.tensor([10,36])).sum(-1).squeeze(-1),(0, max_len - len(src) + 2, 0, max_len - len(src) + 2), "constant", 0)
        relationship_9 = F.pad(forest.softmax(-1).index_select(dim=-1,index = torch.tensor([2])).sum(-1).squeeze(-1),(0, max_len - len(src) + 2, 0, max_len - len(src) + 2), "constant", 0)
        relationship_10 = F.pad(forest.softmax(-1).index_select(dim=-1,index = torch.tensor([4])).sum(-1).squeeze(-1),(0, max_len - len(src) + 2, 0, max_len - len(src) + 2), "constant", 0)
        relationship_11 = F.pad(forest.softmax(-1).index_select(dim=-1,index = torch.tensor([3])).sum(-1).squeeze(-1),(0, max_len - len(src) + 2, 0, max_len - len(src) + 2), "constant", 0)
        relationship_12 = F.pad(forest.softmax(-1).index_select(dim=-1,index = torch.tensor([25,44])).sum(-1).squeeze(-1),(0, max_len - len(src) + 2, 0, max_len - len(src) + 2), "constant", 0)
        relationship_13 = F.pad(forest.softmax(-1).index_select(dim=-1,index = torch.tensor([15,37])).sum(-1).squeeze(-1),(0, max_len - len(src) + 2, 0, max_len - len(src) + 2), "constant", 0)
        relationship_14 = F.pad(forest.softmax(-1).index_select(dim=-1,index = torch.tensor([28,29,41])).sum(-1).squeeze(-1),(0, max_len - len(src) + 2, 0, max_len - len(src) + 2), "constant", 0)
        relationship_15 = F.pad(forest.softmax(-1).index_select(dim=-1,index = torch.tensor([5])).sum(-1).squeeze(-1),(0, max_len - len(src) + 2, 0, max_len - len(src) + 2), "constant", 0)
        relationship_16 = F.pad(forest.softmax(-1).index_select(dim=-1,index = torch.tensor([40])).sum(-1).squeeze(-1),(0, max_len - len(src) + 2, 0, max_len - len(src) + 2), "constant", 0)
        total_relationship = torch.stack([relationship_1,relationship_2,relationship_3,relationship_4,relationship_5,relationship_6,relationship_7,
                                              relationship_8,relationship_9,relationship_10,relationship_11,relationship_12,relationship_13,relationship_14,relationship_15,
                                              relationship_16],dim=0)
        total_relationship = torch.nan_to_num(total_relationship)
        final_dataset.append(total_relationship)
    return final_dataset
