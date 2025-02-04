'''
以alpha-quantile概率作为标准来初始化模型分配
使用模型运行的结果作为优化数据集构建的标准
使用RL优化数据集的构建
'''
# -*- coding: utf-8 -*-
import argparse
import copy
import json
import os
import pickle
import random
import re
import sys
import time
from datetime import datetime
from typing import List

import numpy as np
import openai
from groq import Groq
from skopt import gp_minimize  # 原来他的全程叫做: scikit-optimize
from skopt.space import Real
from tqdm import tqdm

sys.path.append('C:\\Users\\Pluto\\Desktop\\TaDe')  # 添加路径方便import
from CHAMP_Trys.CHAMP_utils import *
from utils import *

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
openaiClient = setOpenAi(keyid = 2)
llamaClient = Groq(  # 这个是Groq调用llama的api接口
    api_key='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'
)
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "alpha-quantile-贝叶斯优化搜索_1001"


'''一组超参设置'''
N = 50  # 用于搜索参数的问题数量
MAX_TRY = 7  # 错误尝试最大值
alpha = 0.0  # 文章里使用了0 0.4 0.8三个选项
thres5 = [0.9, 0.8, 0.5, 0.4, 0.2]

# 数据存储路径
file_path = '../Task_Datasets/CHAMP/all_champ_p.json'
with open(file_path, 'r', encoding='utf-8') as file:
    problems = json.load(file)
now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
tokens_path = f'Tokens/token_usage_{formatted_now}.json'  # 这是记录token消耗的文件
if not os.path.exists(tokens_path):
    with open(tokens_path, 'w') as f:
        json.dump({}, f)

with open('CHAMP_config.json', 'r') as f:
    config = json.load(f)
config['tokens_path'] = tokens_path      




def evaluate(thres5):
    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    question_ids = list(range(N))
    '''开始模型分配'''
    depths_intEdges = {}
    for question_id in tqdm(question_ids):
        # allResInfo[question_id] = {}
        depths_intEdges[question_id] = {}
        with open(f'Decomposed_Steps/{question_id}.json', 'r', encoding='utf-8') as f:
            pre_steps = json.load(f)
        
        question = problems[question_id]['problem_text']
        gold_answer = problems[question_id]['problem_answer']
   
        for solve_time in range(2):
            print(f"question_id: {question_id}  solve_time: {solve_time}")
            depths_intEdges[question_id][solve_time] = {}
            steps, steps_dict = pre_steps[solve_time]
            attempts = 0
            success = False
            while attempts < MAX_TRY and not success:  # 如果遇到格式错误
                try:
                    relations_test = construct_dependencies_without_traversal(clients, question, steps, config)  # query LLM回答所有的依赖
                    G1 = create_dag_from_string(relations_test)
                    reduced_dependencies = list(G1.edges())
                    edges = []
                    for item in reduced_dependencies:
                        edges.append((item[0][:item[0].find('[')].strip(), item[1][:item[1].find('[')].strip()))
                    int_edges = [(int(e[0].split()[1]), int(e[1].split()[1])) for e in edges]
                    node_depths = calculate_node_depths(edges)
                    depths = reverseDict(node_depths)
                    
                    depths_intEdges[question_id][solve_time]['depths'] = depths
                    depths_intEdges[question_id][solve_time]['int_edges'] = int_edges
                    

                    # 开始基于图进行推理
                    heights = list(depths.keys())
                    MAXHeight = max(heights)
                    answerDict = {}
                    progress_bar = tqdm(total=len(steps))
                    
                    frac_models = {}
                    sub_probs = {}
                    for i in range(MAXHeight):
                        subtasks = depths[i]
                        for subtaskid in subtasks: 
                            number = re.findall(r'\d+', subtaskid)
                            number = int(number[0]) if number else None
                            subtask = steps_dict[str(number)]
                            answer_MODEL = config["alpha-quantile-base"]  # 用gpt-3.5-turbo做回答
                            
                            result, prob_values, Q = solve_Sub_Question(clients, question, answerDict, int_edges, number, subtask, tokens_path, answer_MODEL, alpha)
                            # # prob_values 是从小到大排列的,而且已经在e上求了指数
                            frac = quantile(prob_values, alpha)  # 前30%小的位置
                            allo_frac = frac2model(frac, thres5)
                            # if allo_frac!=config["alpha-quantile-base"]:
                            #     result, prob_values, Q = solve_Sub_Question(clients, question, answerDict, int_edges, number, subtask, tokens_path, allo_frac, alpha) 
                            answerDict[number] = {'subtask':subtask, 'answer':result}
                            frac_models[number] = allo_frac
                            sub_probs[number] = prob_values
                            progress_bar.update(1)
                                
                    progress_bar.close()

                    # 已经问完了所有的subtask,最后问一次得到最终的答案
                    Q.append({'role':'assistant', 'content':result})
                    Q.append({'role':'user', 'content':f"""Now that all the sub-problems have been solved, so what is the final answer?
Please give the final answer without any additional explanation or clarification."""})
                    # finalResult = askChatGPT(Q, model=config["finalSummarize_MODEL"], temperature=1)
                    finalResult = askLLM(clients, Q, tokens_path=tokens_path, model=config['finalSummarize_MODEL'], temperature=1, max_tokens=300)
                    
                    # 让大语言模型来判断有没有回答正确
                    judgeAnswer = {'role':'user', 'content':f"""Here is a math problem with a standard answer and a student's solution. Please help me determine if the student's solution is correct.
Problem: {question}

Standard answer: {gold_answer}

Answer: {finalResult}

If the student's answer is correct, just output True; otherwise, just output False.
No explanation is required.
"""}
                    Q_judge = [judgeAnswer]
                    ifcorrect = askLLM(clients, Q_judge, tokens_path=tokens_path, model=config['judgeCorrect_MODEL'], temperature=1, max_tokens=300)
                    
                    if 'True' in ifcorrect:
                        success_Q += 1
                        success = True  # 任务未受中断,完整地结束了,所以标记为成功
                    elif 'False' in ifcorrect:
                        unsuccess_Q += 1
                        success = True  # 任务未受中断,完整地结束了,所以标记为成功    
                except:
                    attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
            
            if attempts == MAX_TRY:     # 一律当做错误推理处理
                error_Q += 1

    print(success_Q)  # 50个中的成功了44个
    with open('depths_intEdges.pkl', 'wb') as f:
        pickle.dump(depths_intEdges, f)
    # 返回的是成功率
    return success_Q / N
    


# def evaluate(thres5):
#     success_Q = 0
#     unsuccess_Q = 0
#     error_Q = 0
#     question_ids = list(range(N))
    
#     # allResInfo = {}
#     '''开始模型分配'''
#     depths_intEdges = {}
#     for question_id in tqdm(question_ids):
#         # allResInfo[question_id] = {}
#         depths_intEdges[question_id] = {}
#         with open(f'Decomposed_Steps/{question_id}.json', 'r', encoding='utf-8') as f:
#             pre_steps = json.load(f)
        
#         question = problems[question_id]['problem_text']
#         gold_answer = problems[question_id]['problem_answer']
   
#         for solve_time in range(2):
#             print(f"question_id: {question_id}  solve_time: {solve_time}")
#             depths_intEdges[question_id][solve_time] = {}
#             steps, steps_dict = pre_steps[solve_time]
#             attempts = 0
#             success = False
#             while attempts < MAX_TRY and not success:  # 如果遇到格式错误
#                 try:
#                     relations_test = construct_dependencies_without_traversal(clients, question, steps, config)  # query LLM回答所有的依赖
#                     G1 = create_dag_from_string(relations_test)
#                     reduced_dependencies = list(G1.edges())
#                     edges = []
#                     for item in reduced_dependencies:
#                         edges.append((item[0][:item[0].find('[')].strip(), item[1][:item[1].find('[')].strip()))
#                     int_edges = [(int(e[0].split()[1]), int(e[1].split()[1])) for e in edges]
#                     node_depths = calculate_node_depths(edges)
#                     depths = reverseDict(node_depths)
                    
#                     depths_intEdges[question_id][solve_time]['depths'] = depths
#                     depths_intEdges[question_id][solve_time]['int_edges'] = int_edges
                    

#                     # 开始基于图进行推理
#                     heights = list(depths.keys())
#                     MAXHeight = max(heights)
#                     answerDict = {}
#                     progress_bar = tqdm(total=len(steps))
                    
#                     frac_models = {}
#                     sub_probs = {}
#                     for i in range(MAXHeight):
#                         subtasks = depths[i]
#                         for subtaskid in subtasks: 
#                             number = re.findall(r'\d+', subtaskid)
#                             number = int(number[0]) if number else None
#                             subtask = steps_dict[str(number)]
#                             answer_MODEL = config["alpha-quantile-base"]  # 用gpt-3.5-turbo做回答
                            
#                             result, prob_values, Q = solve_Sub_Question(clients, question, answerDict, int_edges, number, subtask, tokens_path, answer_MODEL, alpha)
#                             # # prob_values 是从小到大排列的,而且已经在e上求了指数
#                             frac = quantile(prob_values, alpha)  # 前30%小的位置
#                             allo_frac = frac2model(frac, thres5)
#                             # if allo_frac!=config["alpha-quantile-base"]:
#                             #     result, prob_values, Q = solve_Sub_Question(clients, question, answerDict, int_edges, number, subtask, tokens_path, allo_frac, alpha) 
#                             answerDict[number] = {'subtask':subtask, 'answer':result}
#                             frac_models[number] = allo_frac
#                             sub_probs[number] = prob_values
#                             progress_bar.update(1)
                                
#                     progress_bar.close()

#                     # 已经问完了所有的subtask,最后问一次得到最终的答案
#                     Q.append({'role':'assistant', 'content':result})
#                     Q.append({'role':'user', 'content':f"""Now that all the sub-problems have been solved, so what is the final answer?
# Please give the final answer without any additional explanation or clarification."""})
#                     # finalResult = askChatGPT(Q, model=config["finalSummarize_MODEL"], temperature=1)
#                     finalResult = askLLM(clients, Q, tokens_path=tokens_path, model=config['finalSummarize_MODEL'], temperature=1, max_tokens=300)
                    
#                     # 让大语言模型来判断有没有回答正确
#                     judgeAnswer = {'role':'user', 'content':f"""Here is a math problem with a standard answer and a student's solution. Please help me determine if the student's solution is correct.
# Problem: {question}

# Standard answer: {gold_answer}

# Answer: {finalResult}

# If the student's answer is correct, just output True; otherwise, just output False.
# No explanation is required.
# """}
#                     Q_judge = [judgeAnswer]
#                     ifcorrect = askLLM(clients, Q_judge, tokens_path=tokens_path, model=config['judgeCorrect_MODEL'], temperature=1, max_tokens=300)
                    
#                     if 'True' in ifcorrect:
#                         success_Q += 1
#                         success = True  # 任务未受中断,完整地结束了,所以标记为成功
#                     elif 'False' in ifcorrect:
#                         unsuccess_Q += 1
#                         success = True
#                 except:
#                     attempts += 1
            
#             if attempts == MAX_TRY:
#                 error_Q += 1
#     print(success_Q)
#     with open('depths_intEdges.pkl', 'wb') as f:
#         pickle.dump(depths_intEdges, f)
#     return success_Q / N
    
    

# # 假设 evaluate 是一个计算任务成功率的函数，返回成功率
# def evaluate(thres):
#     # 模拟一个复杂函数，通过超参数组合返回成功率
#     # 你可以根据你的实际任务进行实现
#     return 1 - ((thres[0] - 0.85) ** 2 + (thres[1] - 0.75) ** 2 + (thres[2] - 0.5) ** 2 + 
#                 (thres[3] - 0.35) ** 2 + (thres[4] - 0.15) ** 2)

# # 定义搜索的超参数空间
# space = [
#     Real(0.0, 1.0, name='thres1'),
#     Real(0.0, 1.0, name='thres2'),
#     Real(0.0, 1.0, name='thres3'),
#     Real(0.0, 1.0, name='thres4'),
#     Real(0.0, 1.0, name='thres5'),
# ]

# # 使用 gp_minimize 进行贝叶斯优化
# result = gp_minimize(
#     func=lambda x: -evaluate(x),  # 最大化成功率，因此我们最小化其负数
#     dimensions=space,             # 参数空间
#     n_calls=50,                   # 评估50次
#     random_state=42,              # 固定随机种子，便于复现
#     n_initial_points=10           # 初始采样10个点
# )

# # 输出优化结果
# print("Best success rate:", -result.fun)
# print("Best parameters:", result.x)


if __name__ == '__main__':
    evaluate(thres5)