# 面向MATH数据集的数据集构建过程

import json
import os
import sys

sys.path.append('C:\\Users\\Pluto\\Desktop\\TaDe')
from CHAMP_Trys.CHAMP_utils import *

# model_mapping = {
#     'gpt-4-turbo': 5,
#     'gpt-4': 4,
#     'gpt-4o-mini': 3,
#     'gpt-3.5-turbo': 2,
#     'llama3-70b': 1,
#     'llama3-8b': 0
# }
model_mapping = {
    'gpt-4-turbo': 1,
    'llama3-8b': 0
}


# def read_and_process_file(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#     last_true_dict = None
#     first_false_dict = None
    
#     for line in lines:
#         # 分割模型分配和结果
#         try:
#             model_assignment, result = line.rsplit(' ', 1)
#         except:
#             print(file_path)
#         result = result.strip() == 'True'
        
#         # 解析字典字符串
#         model_dict = eval(model_assignment)

#         if result:
#             last_true_dict = model_dict  # 更新最后一个True行的分配方法
#         else:
#             if first_false_dict is None:  # 只记录第一个False行
#                 first_false_dict = model_dict
#     res = last_true_dict if last_true_dict is not None else first_false_dict
#     if res is None:
#         print(file_path)
#         sys.exit(0)
        
#     # 根据条件返回结果
#     return last_true_dict if last_true_dict is not None else first_false_dict


def read_and_process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()  # 读取所有行
    
    if not lines:
        print(f"The file {file_path} is empty.")
        return None
    
    iftrue = 0
    for l in lines:
        if 'True' in l:
            iftrue = 1

    return len(lines), iftrue  # 直接返回最后一行的字典


def count_unique_problem_ids(folder_path):
    unique_problem_ids = set()
    file_count = 0
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            file_count += 1  # 统计文件数量
            # 假设文件名格式为 problemid_solveid
            parts = filename.split('_')
            if len(parts) > 1:  # 确保文件名格式正确
                problemid = parts[0]  # 获取 problemid
                unique_problem_ids.add(problemid)  # 添加到集合中
            # 在这里读取文件内容
            

    # 返回不同的 problemid 数量和文件数量
    return len(unique_problem_ids), file_count



def Gen_Dataset(folder_path, initial_problems):
    unique_problem_ids = set()
    file_count = 0
    
    QA = []
    Allinfo = []
    
    countfile = 0
    count_length = 0
    count_true = 0
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            file_count += 1  # 统计文件数量
            # 假设文件名格式为 problemid_solveid
            name_without_ext = filename.split('.')[0]
            parts = name_without_ext.split('_')
            if len(parts) > 1:  # 确保文件名格式正确
                problemid = int(parts[0])  # 获取 problemid
                unique_problem_ids.add(problemid)  # 添加到集合中
                
            # 在这里读取文件内容
            length, iftrue = read_and_process_file(os.path.join(folder_path, filename))  
            countfile += 1
            count_length += length
            count_true += iftrue
    return countfile, count_length, count_true



if __name__ == '__main__':
    file_path = '../Task_Datasets/MATH/all_math_p.json'  # 完了这里的数据名称不对
    with open(file_path, 'r', encoding='utf-8') as file:
        problems = json.load(file)
    
    folder_path = './ModelAllocation/Alpha-search'  # 替换为实际路径
    countfile, count_length, count_true = Gen_Dataset(folder_path, problems)
    print(f'平均搜索次数: {count_length/countfile}')
    print(f'平均成功率: {count_true/countfile}')
    
    
    

