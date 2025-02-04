'''
用来判断子模型的模型阶数下降了多少
'''

import json
import os
import sys

sys.path.append('C:\\Users\\Pluto\\Desktop\\TaDe')
from CHAMP_Trys.CHAMP_utils import *

model_mapping = {
    'gpt-4-turbo': 5,
    'gpt-4': 4,
    'gpt-4o-mini': 3,
    'gpt-3.5-turbo': 2,
    'llama3-70b': 1,
    'llama3-8b': 0
}


def read_and_process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    last_true_dict = None
    first_false_dict = None
    ifFirst = True
    initRes = None
    
    for line in lines:
        # 分割模型分配和结果
        model_assignment, result = line.rsplit(' ', 1)
        result = result.strip() == 'True'
        
        # 解析字典字符串
        model_dict = eval(model_assignment)

        if result:
            last_true_dict = model_dict  # 更新最后一个True行的分配方法
        else:
            if first_false_dict is None:  # 只记录第一个False行
                first_false_dict = model_dict
        if ifFirst:
            initRes = result
            ifFirst = False
        
    res = last_true_dict if last_true_dict is not None else first_false_dict
    if res is None:
        print(file_path)
        sys.exit(0)
        
    # 根据条件返回结果
    return last_true_dict if last_true_dict is not None else first_false_dict, initRes


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
    numSub = 0
    numDown = 0
    numDownNo = 0
    
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
            best_allocation, initRes = read_and_process_file(os.path.join(folder_path, filename))
            # print(f"select_dict: {best_allocation}")
            
            
            difficulty_eval = {k: model_mapping[v] for k, v in best_allocation.items()}
            # print(f"difficulty_eval: {difficulty_eval}")
            difficulties = list(difficulty_eval.values())

            if initRes == True:
                numSub += len(difficulties)
                numDown += len(difficulties)*5-sum(difficulties)
                numDownNo += sum(1 for x in difficulties if x < 5)
            

    # 返回不同的 problemid 数量和文件数量
    return numSub, numDown, numDownNo


if __name__ == '__main__':
    file_path = '../Task_Datasets/CHAMP/all_champ_p.json'  # 完了这里的数据名称不对
    with open(file_path, 'r', encoding='utf-8') as file:
        problems = json.load(file)
    
    folder_path = 'Allo_search'  # 替换为实际路径
    numSub, numDown, numDownNo = Gen_Dataset(folder_path, problems)
    print(f"#子任务: {numSub}")
    print(f"#经过优化的子任务: {numDownNo}")
    print(f"#优化子任务占比: {numDownNo/numSub}")
    print(f"#优化平均幅度: {numDown/numDownNo}")



