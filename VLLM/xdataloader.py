import json
import re
import sys
import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from xutils import *
from xmodule import *

import itertools

'''batch里的直接就是encode之后的tensor了'''
class QADataset1(Dataset):
    def __init__(self, startRatio = 0, endRatio = 0.7, p_batch_size = 256, cacheName = 'dataset1_cache.pt'):
        file_path = './QA_dataset/CHAMP/QA-plus_CHAMP_Dataset_4finetuning.json'
        with open(file_path, 'r', encoding='utf-8') as file:
            train_data = json.load(file)
        train_data = train_data
        
        self.problemText = [item["problemText"] for item in train_data]
        self.allSubtask = [item["allSubtask"] for item in train_data]
        self.nowSubtask = [item["nowSubtask"] for item in train_data]
        self.queryTexts = [item["queryText"] for item in train_data]
        self.difficultyNums = [item["difficultyNum"] for item in train_data]
        self.p_batch_size = p_batch_size

        # 预处理所有prompts
        self.queries = self.queryTexts

        if os.path.exists(os.path.join('./QA_dataset/CHAMP/cache', cacheName)):
            self.encoded_prompts = torch.load(os.path.join('./QA_dataset/CHAMP/cache', cacheName))
        else:
            # 预处理文本数据,先保存原始的 is_embedding_model 方法，以便之后需要时可以恢复
            original_is_embedding_model = ModelRegistry.is_embedding_model
            # 将 ModelRegistry 类中的 is_embedding_model 方法替换为 always_true_is_embedding_model
            ModelRegistry.is_embedding_model = always_true_is_embedding_model
            # 现在调用 ModelRegistry.is_embedding_model 无论如何都会返回 True
            # print(ModelRegistry.is_embedding_model("any_model_architecture"))  # 输出 True
            ModelRegistry.register_model("MyLlamaEmbeddingModel", MyLlamaEmbeddingModel)
            emb_LLM = LLM(model="./Models/Meta-Llama-3-8B-Instruct", enforce_eager=True)  # dtype="float16"
            self.emb_LLM = emb_LLM
            self.encoded_prompts = self.process_prompts_in_batches(cacheName)
            # self.encoded_prompts = torch.tensor(
            #     [self.emb_LLM.encode(prompt)[0].outputs.embedding for prompt in self.queries],
            #     dtype=torch.float32
            # )
        length = len(self.difficultyNums)
        self.encoded_prompts = self.encoded_prompts[int(length*startRatio):int(length*endRatio)]
        self.difficultyNums = self.difficultyNums[int(length*startRatio):int(length*endRatio)]

    def process_prompts_in_batches(self, cacheName):
        encoded_list = []
        for i in range(0, len(self.queries), self.p_batch_size):
            batch_prompts = self.queries[i:i + self.p_batch_size]
            encoded_outputs = self.emb_LLM.encode(batch_prompts)  # 一次性编码
            encoded_list.extend([output.outputs.embedding for output in encoded_outputs]) 
        allEmbs = torch.tensor(encoded_list, dtype=torch.float32) 
        save_tensor(allEmbs, './QA_dataset/CHAMP/cache', cacheName)  
        return allEmbs
        
    def __len__(self):
        return len(self.difficultyNums)

    def __getitem__(self, idx):
        '''输入是对整个prompts编码好的embedding'''
        return self.encoded_prompts[idx], self.difficultyNums[idx]

'''
collate_fn 来处理不同长度的序列和答案
需要配合version 1来使用
'''
def collate_fn1(batch): 
    embeddings, answers = zip(*batch)
    
    embeddings = torch.stack(embeddings)
    answers = torch.tensor(answers, dtype=torch.long)
    
    # answers 需要进一步处理，比如编码成序列化数据，这里保持为原字符串
    return embeddings.cuda(), answers.cuda()

# 示例数据加载器
def get_dataloader1(dataset, batch_size=4):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn1, shuffle=True)


'''batch里的直接就是encode之后的tensor了'''
class QADataset2(Dataset):
    def __init__(self, startRatio = 0, endRatio = 0.7, p_batch_size = 256, cacheName = 'dataset2_cache.pt'):
        file_path = './QA_dataset/CHAMP/QA-plus_CHAMP_Dataset_4finetuning.json'
        with open(file_path, 'r', encoding='utf-8') as file:
            train_data = json.load(file)
        train_data = train_data
        
        self.problemText = [item["problemText"] for item in train_data]
        self.allSubtask = [item["allSubtask"] for item in train_data]
        self.nowSubtask = [item["nowSubtask"] for item in train_data]
        self.queryTexts = [item["queryText"] for item in train_data]
        self.difficultyNums = [item["difficultyNum"] for item in train_data]
        self.p_batch_size = p_batch_size

        # 预处理所有prompts
        self.queries = self.queryTexts

        if os.path.exists(os.path.join('./QA_dataset/CHAMP/cache', cacheName)):
            self.emb_tasks, self.emb_subtasks = torch.load(os.path.join('./QA_dataset/CHAMP/cache', cacheName))
        else:
            # 预处理文本数据,先保存原始的 is_embedding_model 方法，以便之后需要时可以恢复
            original_is_embedding_model = ModelRegistry.is_embedding_model
            # 将 ModelRegistry 类中的 is_embedding_model 方法替换为 always_true_is_embedding_model
            ModelRegistry.is_embedding_model = always_true_is_embedding_model
            # 现在调用 ModelRegistry.is_embedding_model 无论如何都会返回 True
            # print(ModelRegistry.is_embedding_model("any_model_architecture"))  # 输出 True
            ModelRegistry.register_model("MyLlamaEmbeddingModel", MyLlamaEmbeddingModel)
            emb_LLM = LLM(model="./Models/Meta-Llama-3-8B-Instruct", enforce_eager=True)  # dtype="float16"
            self.emb_LLM = emb_LLM
            self.emb_tasks, self.emb_subtasks = self.process_prompts_in_batches(cacheName)
        length = len(self.difficultyNums)
        self.emb_tasks = self.emb_tasks[int(length*startRatio):int(length*endRatio)]
        self.emb_subtasks = self.emb_subtasks[int(length*startRatio):int(length*endRatio)]
        self.difficultyNums = self.difficultyNums[int(length*startRatio):int(length*endRatio)]

    def process_prompts_in_batches(self, cacheName):
        emb_tasks = []
        emb_subtasks = []
        for i in range(0, len(self.queries), self.p_batch_size):
            batch_tasks = self.problemText[i:i + self.p_batch_size]
            encoded_tasks = self.emb_LLM.encode(batch_tasks)  # 一次性编码
            emb_tasks.extend([output.outputs.embedding for output in encoded_tasks]) 
            
            batch_subtasks = self.nowSubtask[i:i + self.p_batch_size]
            encoded_subtasks = self.emb_LLM.encode(batch_subtasks)  # 一次性编码
            emb_subtasks.extend([output.outputs.embedding for output in encoded_subtasks])             
            
        allTaskEmbs = torch.tensor(emb_tasks, dtype=torch.float32) 
        allSubtaskEmbs = torch.tensor(emb_subtasks, dtype=torch.float32)
        save_tensor([allTaskEmbs, allSubtaskEmbs], './QA_dataset/CHAMP/cache', cacheName)  
        return allTaskEmbs, allSubtaskEmbs
        
    def __len__(self):
        return len(self.difficultyNums)

    def __getitem__(self, idx):
        '''输入是对整个prompts编码好的embedding'''
        return self.emb_tasks[idx], self.emb_subtasks[idx], self.difficultyNums[idx]

'''
collate_fn 来处理不同长度的序列和答案
需要配合version 1来使用
'''
def collate_fn2(batch): 
    emb_tasks, emb_subtasks, answers = zip(*batch)
    
    s_emb_tasks = torch.stack(emb_tasks)
    s_emb_subtasks = torch.stack(emb_subtasks)
    answers = torch.tensor(answers, dtype=torch.long)
    
    # answers 需要进一步处理，比如编码成序列化数据，这里保持为原字符串
    return s_emb_tasks.cuda(), s_emb_subtasks.cuda(), answers.cuda()

# 示例数据加载器
def get_dataloader2(dataset, batch_size=4):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn2, shuffle=True)



'''
第三种模型构建
输入的是3个embedding
'''
class QADataset3(Dataset):
    def __init__(self, benchmark = 'CHAMP', startRatio = 0, endRatio = 0.7, p_batch_size = 256, cacheName = 'dataset3_cache.pt'):
        file_path = f'./QA_dataset/{benchmark}/QA-1008_{benchmark}_Dataset_4finetuning.json'
        with open(file_path, 'r', encoding='utf-8') as file:
            train_data = json.load(file)
        train_data = train_data
        
        self.problemText = [item["problemText"] for item in train_data]
        self.allSubtask = [item["allSubtask"] for item in train_data]
        self.nowSubtask = [item["nowSubtask"] for item in train_data]
        self.queryTexts = [item["queryText"] for item in train_data]
        self.difficultyNums = [item["difficultyNum"] for item in train_data]
        self.p_batch_size = p_batch_size
        self.queries = self.queryTexts

        if os.path.exists(os.path.join(f'./QA_dataset/{benchmark}/cache', cacheName)):
            self.emb_tasks, self.emb_allsubtasks, self.emb_nowsubtasks = torch.load(os.path.join(f'./QA_dataset/{benchmark}/cache', cacheName))
        else:
            # 预处理文本数据,先保存原始的 is_embedding_model 方法，以便之后需要时可以恢复
            original_is_embedding_model = ModelRegistry.is_embedding_model
            # 将 ModelRegistry 类中的 is_embedding_model 方法替换为 always_true_is_embedding_model
            ModelRegistry.is_embedding_model = always_true_is_embedding_model
            # 现在调用 ModelRegistry.is_embedding_model 无论如何都会返回 True
            # print(ModelRegistry.is_embedding_model("any_model_architecture"))  # 输出 True
            ModelRegistry.register_model("MyLlamaEmbeddingModel", MyLlamaEmbeddingModel)
            emb_LLM = LLM(model="./Models/Meta-Llama-3-8B-Instruct", enforce_eager=True)  # dtype="float16"
            self.emb_LLM = emb_LLM
            self.emb_tasks, self.emb_allsubtasks, self.emb_nowsubtasks = self.process_prompts_in_batches(benchmark, cacheName)
            print(f'Dataset [{benchmark}] tokenization successful!')
            # self.encoded_prompts = torch.tensor(
            #     [self.emb_LLM.encode(prompt)[0].outputs.embedding for prompt in self.queries],
            #     dtype=torch.float32
            # )
            
        num_zeros = self.difficultyNums.count(0)
        num_ones = self.difficultyNums.count(1)

        print(f'0的数量: {num_zeros}')
        print(f'1的数量: {num_ones}')

        length = len(self.difficultyNums)
        self.emb_tasks = self.emb_tasks[int(length*startRatio):int(length*endRatio)]
        self.emb_allsubtasks = self.emb_allsubtasks[int(length*startRatio):int(length*endRatio)]
        self.emb_nowsubtasks = self.emb_nowsubtasks[int(length*startRatio):int(length*endRatio)]
        self.difficultyNums = self.difficultyNums[int(length*startRatio):int(length*endRatio)]

    def process_prompts_in_batches(self, benchmark, cacheName):
        emb_tasks = []
        emb_allsubtasks = []
        emb_nowsubtasks = []
        for i in range(0, len(self.queries), self.p_batch_size):
            batch_tasks = self.problemText[i:i + self.p_batch_size]
            encoded_tasks = self.emb_LLM.encode(batch_tasks)  # 一次性编码
            emb_tasks.extend([output.outputs.embedding for output in encoded_tasks]) 
            
            batch_nowsubtasks = self.nowSubtask[i:i + self.p_batch_size]
            encoded_nowsubtasks = self.emb_LLM.encode(batch_nowsubtasks)  # 一次性编码
            emb_nowsubtasks.extend([output.outputs.embedding for output in encoded_nowsubtasks])   
            
            batch_allsubtasks = self.allSubtask[i:i + self.p_batch_size]
            encoded_allsubtasks = self.emb_LLM.encode(batch_allsubtasks)  # 一次性编码
            emb_allsubtasks.extend([output.outputs.embedding for output in encoded_allsubtasks])             
            
        allTaskEmbs = torch.tensor(emb_tasks, dtype=torch.float32) 
        allallSubtaskEmbs = torch.tensor(emb_allsubtasks, dtype=torch.float32)
        allnowSubtaskEmbs = torch.tensor(emb_nowsubtasks, dtype=torch.float32)
        save_tensor([allTaskEmbs, allallSubtaskEmbs, allnowSubtaskEmbs], f'./QA_dataset/{benchmark}/cache', cacheName)  
        return allTaskEmbs, allallSubtaskEmbs, allnowSubtaskEmbs
        
    def __len__(self):
        return len(self.difficultyNums)

    def __getitem__(self, idx):
        '''输入是对整个prompts编码好的embedding'''
        return self.emb_tasks[idx], self.emb_allsubtasks[idx], self.emb_nowsubtasks[idx], self.difficultyNums[idx]

'''
collate_fn 来处理不同长度的序列和答案
需要配合version 1来使用
'''
def collate_fn3(batch): 
    emb_tasks, emb_allsubtasks, emb_nowsubtasks, answers = zip(*batch)
    
    s_emb_tasks = torch.stack(emb_tasks)
    s_emb_allsubtasks = torch.stack(emb_allsubtasks)
    s_emb_nowsubtasks = torch.stack(emb_nowsubtasks)
    answers = torch.tensor(answers, dtype=torch.long)
    
    # answers 需要进一步处理，比如编码成序列化数据，这里保持为原字符串
    return s_emb_tasks.cuda(), s_emb_allsubtasks.cuda(), s_emb_nowsubtasks.cuda(), answers.cuda()

# 示例数据加载器
def get_dataloader3(dataset, batch_size=4):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn3, shuffle=True, drop_last=True)




'''
第三种模型构建
输入的是3个embedding
'''
class QADataset4(Dataset):
    def __init__(self, startRatio = 0, endRatio = 0.7, p_batch_size = 256, cacheName = 'dataset4_cache.pt'):
        file_path = './QA_dataset/CHAMP/QA-plus_CHAMP_Dataset_4finetuning.json'
        with open(file_path, 'r', encoding='utf-8') as file:
            train_data = json.load(file)
        train_data = train_data
        
        self.problemText = [item["problemText"] for item in train_data]
        self.allSubtask = [item["allSubtask"] for item in train_data]
        self.nowSubtask = [item["nowSubtask"] for item in train_data]
        self.queryTexts = [item["queryText"] for item in train_data]
        self.difficultyNums = [item["difficultyNum"] for item in train_data]
        self.p_batch_size = p_batch_size

        # 预处理所有prompts
        self.queries = self.queryTexts

        if os.path.exists(os.path.join('./QA_dataset/CHAMP/cache', cacheName)):
            self.emb_allsubtasks, self.emb_nowsubtasks = torch.load(os.path.join('./QA_dataset/CHAMP/cache', cacheName))
        else:
            # 预处理文本数据,先保存原始的 is_embedding_model 方法，以便之后需要时可以恢复
            original_is_embedding_model = ModelRegistry.is_embedding_model
            # 将 ModelRegistry 类中的 is_embedding_model 方法替换为 always_true_is_embedding_model
            ModelRegistry.is_embedding_model = always_true_is_embedding_model
            # 现在调用 ModelRegistry.is_embedding_model 无论如何都会返回 True
            # print(ModelRegistry.is_embedding_model("any_model_architecture"))  # 输出 True
            ModelRegistry.register_model("MyLlamaEmbeddingModel", MyLlamaEmbeddingModel)
            emb_LLM = LLM(model="./Models/Meta-Llama-3-8B-Instruct", enforce_eager=True)  # dtype="float16"
            self.emb_LLM = emb_LLM
            self.emb_allsubtasks, self.emb_nowsubtasks = self.process_prompts_in_batches(cacheName)
        length = len(self.difficultyNums)
        self.emb_allsubtasks = self.emb_allsubtasks[int(length*startRatio):int(length*endRatio)]
        self.emb_nowsubtasks = self.emb_nowsubtasks[int(length*startRatio):int(length*endRatio)]
        self.difficultyNums = self.difficultyNums[int(length*startRatio):int(length*endRatio)]

    def process_prompts_in_batches(self, cacheName):
        emb_allsubtasks = []
        emb_nowsubtasks = []
        for i in range(0, len(self.queries), self.p_batch_size):
            batch_tasks = self.problemText[i:i + self.p_batch_size]
            encoded_tasks = self.emb_LLM.encode(batch_tasks)  # 一次性编码
            
            batch_nowsubtasks = self.nowSubtask[i:i + self.p_batch_size]
            encoded_nowsubtasks = self.emb_LLM.encode(batch_nowsubtasks)  # 一次性编码
            emb_nowsubtasks.extend([output.outputs.embedding for output in encoded_nowsubtasks])   
            
            batch_allsubtasks = self.allSubtask[i:i + self.p_batch_size]
            encoded_allsubtasks = self.emb_LLM.encode(batch_allsubtasks)  # 一次性编码
            emb_allsubtasks.extend([output.outputs.embedding for output in encoded_allsubtasks])             
            
        allallSubtaskEmbs = torch.tensor(emb_allsubtasks, dtype=torch.float32)
        allnowSubtaskEmbs = torch.tensor(emb_nowsubtasks, dtype=torch.float32)
        save_tensor([allallSubtaskEmbs, allnowSubtaskEmbs], './QA_dataset/CHAMP/cache', cacheName)  
        return allallSubtaskEmbs, allnowSubtaskEmbs
        
    def __len__(self):
        return len(self.difficultyNums)

    def __getitem__(self, idx):
        '''输入是对整个prompts编码好的embedding'''
        return self.emb_allsubtasks[idx], self.emb_nowsubtasks[idx], self.difficultyNums[idx]

'''
collate_fn 来处理不同长度的序列和答案
需要配合version 1来使用
'''
def collate_fn4(batch): 
    emb_allsubtasks, emb_nowsubtasks, answers = zip(*batch)
    
    s_emb_allsubtasks = torch.stack(emb_allsubtasks)
    s_emb_nowsubtasks = torch.stack(emb_nowsubtasks)
    answers = torch.tensor(answers, dtype=torch.long)
    
    # answers 需要进一步处理，比如编码成序列化数据，这里保持为原字符串
    return s_emb_allsubtasks.cuda(), s_emb_nowsubtasks.cuda(), answers.cuda()




# 示例数据加载器
def get_dataloader4(dataset, batch_size=4):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn4, shuffle=True)

dataset_mapping = {
    1: QADataset1,
    2: QADataset2,
    3: QADataset3,
    4: QADataset4
}

dataloader_mapping = {
    1: get_dataloader1,
    2: get_dataloader2,
    3: get_dataloader3,
    4: get_dataloader4
}

# 函数来选择指定的数据集和dataloader
def select_dataset_and_loader(choice, Ratio=0.7, batch_size=32):
    if choice not in dataset_mapping or choice not in dataloader_mapping:
        raise ValueError("Invalid choice. Please choose a number between 1 and 4.")

    # 选择指定的数据集和dataloader
    dataset_class = dataset_mapping[choice]
    dataloader_func = dataloader_mapping[choice]
    
    # 创建训练集和验证集
    train_dataset = dataset_class(startRatio=0, endRatio=Ratio)
    train_loader = dataloader_func(train_dataset, batch_size=batch_size)

    eval_dataset = dataset_class(startRatio=Ratio, endRatio=1)
    eval_loader = dataloader_func(eval_dataset, batch_size=batch_size)

    return train_loader, eval_loader



'''
专门在eval阶段,也就是实际模型分配的阶段使用
'''
class QADatasetApplication(Dataset):
    def __init__(self, dataset, p_batch_size = 256, cacheName = 'dataset3_cache.pt'):
        # dataset['0'].keys():    ['steps', 'steps_dict', 'depths', 'int_edges', 'problemText', 'allSubtask', 'nowSubtask']
        
        self.problemText = [item["problemText"] for item in dataset.values()]
        self.allSubtask = [item["allSubtask"] for item in dataset.values()]
        self.nowSubtask = [item["nowSubtask"] for item in dataset.values()]  # 这个有点问题
        self.p_batch_size = p_batch_size
        
        # 需要根据self.nowSubtask的list长度来扩展前两个list
        self.problemText = [self.problemText[i] for i in range(len(self.problemText)) for _ in range(len(self.nowSubtask[i]))]
        self.allSubtask = [self.allSubtask[i] for i in range(len(self.allSubtask)) for _ in range(len(self.nowSubtask[i]))]
        # 使用 itertools.chain 将所有子列表拼接成一个扁平化的列表
        self.nowSubtask = list(itertools.chain(*self.nowSubtask))
        
        
        if os.path.exists(os.path.join(f'./QA_dataset/Apply/cache', cacheName)):
            self.emb_tasks, self.emb_allsubtasks, self.emb_nowsubtasks = torch.load(os.path.join(f'./QA_dataset/Apply/cache', cacheName))
        else:
            # 预处理文本数据,先保存原始的 is_embedding_model 方法，以便之后需要时可以恢复
            original_is_embedding_model = ModelRegistry.is_embedding_model
            # 将 ModelRegistry 类中的 is_embedding_model 方法替换为 always_true_is_embedding_model
            ModelRegistry.is_embedding_model = always_true_is_embedding_model
            # 现在调用 ModelRegistry.is_embedding_model 无论如何都会返回 True
            # print(ModelRegistry.is_embedding_model("any_model_architecture"))  # 输出 True
            ModelRegistry.register_model("MyLlamaEmbeddingModel", MyLlamaEmbeddingModel)
            emb_LLM = LLM(model="./Models/Meta-Llama-3-8B-Instruct", enforce_eager=True)  # dtype="float16"
            self.emb_LLM = emb_LLM
            self.emb_tasks, self.emb_allsubtasks, self.emb_nowsubtasks = self.process_prompts_in_batches(cacheName)
            print(f'Dataset tokenization successful!')


    def process_prompts_in_batches(self, cacheName):
        emb_tasks = []
        emb_allsubtasks = []
        emb_nowsubtasks = []
        for i in range(0, len(self.problemText), self.p_batch_size):
                batch_tasks = self.problemText[i:i + self.p_batch_size]
                encoded_tasks = self.emb_LLM.encode(batch_tasks)  # 一次性编码
                emb_tasks.extend([output.outputs.embedding for output in encoded_tasks]) 
                
                batch_nowsubtasks = self.nowSubtask[i:i + self.p_batch_size]
                encoded_nowsubtasks = self.emb_LLM.encode(batch_nowsubtasks)  # 一次性编码
                emb_nowsubtasks.extend([output.outputs.embedding for output in encoded_nowsubtasks])   
                
                batch_allsubtasks = self.allSubtask[i:i + self.p_batch_size]
                encoded_allsubtasks = self.emb_LLM.encode(batch_allsubtasks)  # 一次性编码
                emb_allsubtasks.extend([output.outputs.embedding for output in encoded_allsubtasks])             
            
        allTaskEmbs = torch.tensor(emb_tasks, dtype=torch.float32) 
        allallSubtaskEmbs = torch.tensor(emb_allsubtasks, dtype=torch.float32)
        allnowSubtaskEmbs = torch.tensor(emb_nowsubtasks, dtype=torch.float32)
        save_tensor([allTaskEmbs, allallSubtaskEmbs, allnowSubtaskEmbs], f'./QA_dataset/Apply/cache', cacheName)  
        return allTaskEmbs, allallSubtaskEmbs, allnowSubtaskEmbs
        
    def __len__(self):
        return len(self.problemText)

    def __getitem__(self, idx):
        '''输入是对整个prompts编码好的embedding'''
        return self.emb_tasks[idx], self.emb_allsubtasks[idx], self.emb_nowsubtasks[idx]

'''
collate_fn 来处理不同长度的序列和答案
需要配合version 1来使用
'''
def collate_fnApplication(batch): 
    emb_tasks, emb_allsubtasks, emb_nowsubtasks = zip(*batch)
    
    s_emb_tasks = torch.stack(emb_tasks)
    s_emb_allsubtasks = torch.stack(emb_allsubtasks)
    s_emb_nowsubtasks = torch.stack(emb_nowsubtasks)
    
    # answers 需要进一步处理，比如编码成序列化数据，这里保持为原字符串
    return s_emb_tasks.cuda(), s_emb_allsubtasks.cuda(), s_emb_nowsubtasks.cuda(),

# 示例数据加载器
def get_dataloaderApplication(dataset):
    return DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fnApplication, shuffle=False, drop_last=True)


