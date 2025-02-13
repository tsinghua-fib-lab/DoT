from vllm import LLM
from vllm.model_executor.models import ModelRegistry
from vllm import LLM, SamplingParams
from vllm.attention import AttentionMetadata
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import PoolerOutput
from vllm import ModelRegistry
from xutils import *
from xdataloader import *

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


import os
import matplotlib.pyplot as plt
from typing import Iterable, List, Optional, Tuple
from tqdm import tqdm

lr = 1e-4
num_epochs = 20000
train_ratio = 0.7  # 使用70%的数据进行训练
method = 3

writer = SummaryWriter(f"logs/train{method}")  # 启动了监听系统.

train_loader, eval_loader = select_dataset_and_loader(method, Ratio=0.7, batch_size=32)
myTaskHead = select_model(f"EmbeddingToScore_v{method}", input_dim=4096, hidden_dim=1024)
total_params = sum(p.numel() for p in myTaskHead.parameters())
print(f'Total number of parameters: {total_params}')



'''-----------模型训练部分-----------'''

# 定义优化器和学习率
optimizer = optim.Adam(myTaskHead.parameters(), lr=lr)
losses = []  # 保存每个epoch的loss

# 创建文件夹
os.makedirs("train_pics", exist_ok=True)

# Training loop
pbar = tqdm(range(num_epochs), ncols=100)
for epoch in range(num_epochs):    
    total_loss = 0
    
    for batch in train_loader:        
        # 前向传播计算 loss
        out = myTaskHead(batch[:-1])  
        # 计算mae loss
        loss = F.l1_loss(out, batch[-1].view(-1, 1))

        # 反向传播并更新参数
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 优化器更新参数
        total_loss += loss.item()            

    average_loss = total_loss / len(train_loader)
    writer.add_scalar('loss', average_loss, epoch)
    losses.append(average_loss)  # 保存每个epoch的loss
    # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}')
    pbar.set_description(f"Epoch {epoch}/{num_epochs}")
    pbar.set_postfix(loss=average_loss)
    pbar.update(1)

save_model(myTaskHead, "Pths", f"v{method}.pth")
writer.close()

# 绘制loss曲线
plt.figure()
plt.plot(range(1, num_epochs + 1), losses, marker='o', markersize=4, color='purple')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')  # 设置y轴为对数坐标
plt.grid()
plt.savefig(os.path.join("train_pics", f"loss_curve_v{method}_lr-{lr}_epoch-{num_epochs}.png"))  # 保存图片
plt.close()  # 关闭当前图形
print("Training finished.")


'''测试过程'''
myTaskHead.eval()  # 设置模型为评估模式
total_absolute_error = 0

with torch.no_grad():  # 禁用梯度计算
    for batch in eval_loader:
        embeddings, golds = batch
        
        # 前向传播
        out = myTaskHead(embeddings)
        
        # 计算绝对误差
        absolute_error = torch.abs(out - golds.view(-1, 1))
        total_absolute_error += absolute_error.sum().item()

average_absolute_error = total_absolute_error / len(train_loader.dataset)
print(f'Average Absolute Error: {average_absolute_error}')


