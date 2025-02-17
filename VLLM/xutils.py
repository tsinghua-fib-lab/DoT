from torch import nn
from vllm import LLM
from vllm.model_executor.models import ModelRegistry
from vllm import LLM, SamplingParams

from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from vllm.attention import AttentionMetadata
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import PoolerOutput
from xutils import *
from xmodule import *
import torch
from torch.utils.data import Dataset, DataLoader
import json
import re
import sys
import os
import logging

# not same as mistral one
class MyLlamaEmbeddingModel(nn.Module):
    """A model that uses Llama with additional embedding functionalities.

   This class encapsulates the LlamaModel and provides an interface for
   embedding operations and customized pooling functions.

   Attributes:
       model: An instance of LlamaModel used for forward operations.
       _pooler: An instance of Pooler used for pooling operations.
   """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = LlamaModel(**kwargs)
        self._pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model.forward(input_ids, positions, kv_caches,
                                  attn_metadata, inputs_embeds)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]): 
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                if name.endswith("kv_scale"):
                    remapped_kv_scale_name = name.replace(
                        ".kv_scale", ".attn.kv_scale")
                    if remapped_kv_scale_name not in params_dict:
                        # print_warning_once(
                        #     f"Found kv scale in the checkpoint (e.g. {name}), "
                        #     "but not found the expected name in the model "
                        #     f"(e.g. {remapped_kv_scale_name}). kv-scale is "
                        #     "not loaded.")
                        continue
                    else:
                        name = remapped_kv_scale_name
                if "lm_head" in name: # 看具体情况 你lm head 可能多了
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

def always_true_is_embedding_model(model_arch: str) -> bool:
    return True

# 创建一个函数来动态选择模型
def select_model(model_name, **kwargs):
    if model_name == "MLP":
        return MLP(**kwargs).to('cuda')
    elif model_name == "EmbeddingToScore_v1":
        return Embedding2Score_v1(**kwargs).to('cuda')
    elif model_name == "EmbeddingToScore_v2":
        return Embedding2Score_v2(**kwargs).to('cuda')
    elif model_name == "EmbeddingToScore_v3":
        return Embedding2Score_v3(**kwargs).to('cuda')
    elif model_name == "EmbeddingToScore_v4":  # 也是2个4096维度的向量拼接
        return Embedding2Score_v2(**kwargs).to('cuda')
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    

def save_model(model, save_dir, file_name):
    """
    保存模型参数到指定文件夹和文件名

    Args:
    - model (nn.Module): 需要保存的PyTorch模型
    - save_dir (str): 保存模型的文件夹路径
    - file_name (str): 保存模型的文件名
    """
    # 创建保存模型的文件夹
    os.makedirs(save_dir, exist_ok=True)
    
    # 定义保存路径
    save_path = os.path.join(save_dir, file_name)
    
    # 保存模型参数
    torch.save(model.state_dict(), save_path)
    
    print(f"\nModel saved to {save_path}")
    
    
def save_tensor(tensor, save_dir, file_name):
    """
    保存 tensor 数据到指定文件夹和文件名

    Args:
    - tensor (torch.Tensor): 需要保存的 tensor 数据
    - save_dir (str): 保存 tensor 的文件夹路径
    - file_name (str): 保存 tensor 的文件名
    """
    # 创建保存文件夹
    os.makedirs(save_dir, exist_ok=True)
    
    # 定义保存路径
    save_path = os.path.join(save_dir, file_name)
    
    # 保存 tensor 数据
    torch.save(tensor, save_path)
    
    print(f"Tensor saved to {save_path}")
    
    
def printSeq(a):
    for i in a:
        print(i)


# 定义模型和数字的映射
model_mapping = {
    5: 'gpt-4-turbo',
    4: 'gpt-4',
    3: 'gpt-4o-mini',
    2: 'gpt-3.5-turbo',
    1: 'llama3-70b',
    0: 'llama3-8b'
}



def write_json_listoneline(file_path, data):
    try:
        # 自定义递归函数，用于处理 list 和其他类型的数据
        def custom_json_encoder(obj, indent=0):
            # 定义缩进
            indent_str = ' ' * indent
            
            if isinstance(obj, dict):
                # 处理 dict 类型
                json_str = '{\n'
                for i, (key, value) in enumerate(obj.items()):
                    if i > 0:
                        json_str += ',\n'
                    json_str += f'{indent_str}  "{key}": {custom_json_encoder(value, indent + 2)}'
                json_str += f'\n{indent_str}}}'
                return json_str

            elif isinstance(obj, list):
                # 处理 list 类型，不换行
                return json.dumps(obj, separators=(',', ':'))

            else:
                # 处理其他类型
                return json.dumps(obj, ensure_ascii=False)

        # 打开文件并写入数据
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(custom_json_encoder(data))
        
        print(f"数据成功写入 {file_path}")
    except Exception as e:
        print(f"发生错误：{e}")
        
    
def setup_logger(tailName=""):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler("logtxt/"+tailName+".log")

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def count_misclassifications(allout, allgold):
    # 初始化计数器
    one_to_zero = 0  # 1 被错误地预测成 0
    zero_to_one = 0  # 0 被错误地预测成 1

    # 遍历两个列表
    for out, gold in zip(allout, allgold):
        if gold == 1 and out == 0:
            one_to_zero += 1  # 统计 1 被错误地预测成 0
        elif gold == 0 and out == 1:
            zero_to_one += 1  # 统计 0 被错误地预测成 1

    # 返回两个统计结果
    return one_to_zero, zero_to_one


