# 面向MATH数据集的数据集构建过程

import json
import os
import sys

sys.path.append('C:\\Users\\Pluto\\Desktop\\TaDe')
from DROP_Trys.DROP_utils import *

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
    
    # 处理最后一行
    last_line = lines[-1]
    
    try:
        model_assignment, result = last_line.rsplit(' ', 1)  # 分割字典和True/False
    except ValueError:
        print(f"Error processing file: {file_path}")
        return None
    
    # 解析字典字符串
    try:
        model_dict = eval(model_assignment)
    except Exception as e:
        print(f"Error parsing the dictionary in file {file_path}: {e}")
        return None
    
    return model_dict  # 直接返回最后一行的字典



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
            best_allocation = read_and_process_file(os.path.join(folder_path, filename))
            # print(f"select_dict: {best_allocation}")
            
            
            difficulty_eval = {k: model_mapping[v] for k, v in best_allocation.items()}
            # print(f"difficulty_eval: {difficulty_eval}")
            difficulties = list(difficulty_eval.values())
            
            # 获取分解好的步骤
            with open(os.path.join('Decomposed_Steps_1004', f'{problemid}.json'), 'r', encoding='utf-8') as file:
                allsteps = json.load(file)
            steps = allsteps[int(parts[1])][0]  # 只需要list格式的就足够了
            # print(f"steps: {steps}")
            
            if len(steps) != len(best_allocation.items()) or len(steps) != len(difficulty_eval.items()):
                print(filename)
                print("Error: The lengths of steps and select_dict do not match!")
                # exit(1)  # 或者使用 sys.exit(1)
                continue
            
            # 串联起来的步骤
            formatted_steps = '; '.join([f'step{i+1}: {step}' for i, step in enumerate(steps)])
                        
            for index, step in enumerate(steps):
            
                Query = f"""You are a sub-question difficulty evaluator. I will provide you with an original question and a sub-question derived from it. I hope you can combine the original question and the current sub-question to give an overall difficulty rating. The rating is a non-negative integer, with a minimum of 0 indicating very easy and a maximum of 5 indicating very difficult.

The context of the question is as follows: {initial_problems[problemid]['passage']}
The original question is as follows: {initial_problems[problemid]['question']}
All sub-tasks are as follows: {formatted_steps}
The sub-task that needs to be evaluated is as follows: {step}

Please conduct the evaluation, and output the format as: score: {{result of your evaluation}}"""
                try:
                    Answer = f"score: {difficulties[index]}"
                except:
                    print(difficulties)
                    print(len(steps))
                    exit(1)
                
                QA.append({'Query':Query, 'Answer':Answer})
                qsa = {
                    "problemText" :initial_problems[problemid]['passage']+'\n'+initial_problems[problemid]['question'], 
                    "allSubtask": formatted_steps, 
                    "nowSubtask": step, 
                    "queryText": Query, 
                    "difficultyNum": difficulties[index],
                }
                Allinfo.append(qsa)
                
            

    # 返回不同的 problemid 数量和文件数量
    return QA, Allinfo


if __name__ == '__main__':
    file_path = '../Task_Datasets/DROP/all_drop_p.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        problems = json.load(file)
    
    folder_path = './ModelAllocation/Alpha-search'  # 替换为实际路径
    QA, Allinfo = Gen_Dataset(folder_path, problems)
    print(len(Allinfo))
    
    write_json(f"QA-1008_DROP_Dataset_4finetuning.json", Allinfo)  



