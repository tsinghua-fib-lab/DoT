import pickle
from xutils import *
from xdataloader import *

import torch
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from collections import Counter

# 输入需要包括三个部分:
# champ, subtask, nowsubtask
# 可以用如下的字典格式来进行存放
# self.problemText = [item["problemText"] for item in train_data]
# self.allSubtask = [item["allSubtask"] for item in train_data]
# self.nowSubtask = [item["nowSubtask"] for item in train_data]

model_mapping = {
    1: 'gpt-4-turbo',
    0: 'llama3-8b'
}


name = "step1Res_scan-last"
name2 = "step2In_scan_last"
pth = "Pths/SCAN/v3_700_minerror.pth"


f = open(f'Input/{name}.json', 'r')
content = f.read()
dataset = json.loads(content)


evaldataset = QADatasetApplication(dataset, cacheName=f'{name}.pt')  # 注意最大序列长度的同步
eval_loader = get_dataloaderApplication(evaldataset)  # 一个batch全部送到网络里面
print(f"len(evaldataset): {len(evaldataset)}")


myTaskHead = select_model("EmbeddingToScore_v3", input_dim=4096, hidden_dim=1024)
state_dict = torch.load(pth)  # 加载保存的模型参数
myTaskHead.load_state_dict(state_dict)

'''测试过程'''
myTaskHead.eval()  # 设置模型为评估模式
total_absolute_error = 0
sigmoid_layer = nn.Sigmoid()
with torch.no_grad():  # 禁用梯度计算
    # 现在只有一个batch了,非常有意思
    for batch in eval_loader:
        taskEmbs, allsubtaskEmbs, nowsubtaskEmbs = batch
        # 前向传播
        out = myTaskHead(taskEmbs, allsubtaskEmbs, nowsubtaskEmbs)
        out = sigmoid_layer(out)
        # print(sum(out)/len(out))
        out = out.cpu().numpy()  # (32, 1)
        out = [1 if x > 0.5 else 0 for x in out.flatten()]


models = [model_mapping[n] for n in out]
# 使用 Counter 统计不同字符串出现的次数
count = Counter(models)
count_dict = dict(count)
print(count_dict)

index = 0
for k, v in dataset.items():
    length = len(v['steps'])
    v['allo_model'] = models[index:index+length]
    index = index+length


write_json_listoneline(f"Output/{name2}.json", dataset)


