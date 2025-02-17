import torch
import torch.nn as nn
import sys

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        """
        初始化 MLP 层，包括一个隐藏层和输出层。

        参数:
        - input_dim (int): 输入的维度
        - output_dim (int): 输出的维度
        - hidden_dim (int): 隐藏层的维度(可选,默认128)
        """
        super(MLP, self).__init__()
        
        # 定义多层感知机的结构
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，接收输入并通过 MLP 计算输出。

        参数:
        - x (torch.Tensor): 输入的张量，形状为 (batch_size, input_dim)

        返回:
        - torch.Tensor: 输出的张量，形状为 (batch_size, output_dim)
        """
        return self.mlp(x)
    

# 简单的多层感知机,完成从embedding到分数的映射。    
class Embedding2Score_v1(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=1024):
        super(Embedding2Score_v1, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x):
        x = self.fc1(x) 
        x = self.relu(x)  
        x = self.fc2(x) 
        x = self.relu(x)  
        score = self.fc3(x)
        return score
    
    
class Embedding2Score_v2(nn.Module):
    '''
    输入为两个4096维的embedding,将它们拼接,形成8192维的向量
    '''
    def __init__(self, input_dim=4096, hidden_dim=1024):
        super(Embedding2Score_v2, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, 1)
        self.relu = nn.ReLU()
    
    def forward(self, embedding1, embedding2):
        x = torch.cat((embedding1, embedding2), dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        score = self.fc3(x)
        return score
    

class Embedding2Score_v3(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=1024):
        super(Embedding2Score_v3, self).__init__()
        # 将3个输入embedding合并为一个向量
        self.fc1 = nn.Linear(input_dim * 3, hidden_dim)  # 第一层，全连接层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)  # 第二层，全连接层
        self.fc3 = nn.Linear(hidden_dim//2, 1)     # 输出层，输出一个分数
        self.relu = nn.ReLU()            # ReLU 激活函数
        # self.sigmoid_layer = nn.Sigmoid()
        

    def forward(self, embedding1, embedding2, embedding3):
        
        # 将3个embedding拼接为一个向量
        x = torch.cat((embedding1, embedding2, embedding3), dim=-1)
        
        # 前向传播
        x = self.relu(self.fc1(x))  # 第一层 + ReLU
        x = self.relu(self.fc2(x))  # 第二层 + ReLU
        score = self.fc3(x)         # 输出层，得到一个实数分数
        # score = self.sigmoid_layer(score)
        return score
    
    
