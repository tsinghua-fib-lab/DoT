'''
根据一定的模型弱化规则逐渐将较大的模型替换为较小的模型
设置了好几个中间的模型分配结果,用于高效搜索模型
好像这个逻辑也没啥问题
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
from tqdm import tqdm

sys.path.append('C:\\Users\\Pluto\\Desktop\\TaDe')
from CHAMP_Trys.CHAMP_utils import *
from utils import *

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
openaiClient = setOpenAi(keyid = 3)
llamaClient = Groq(  # 这个是Groq调用llama的api接口
    api_key='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'
)
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "LLM_allocation_search-1001-multi_middle"  # 及时更新这个日期,需要增大将本地


# 输入一个模型分配的结果,返回True或者是False的结果
def _reason(allo_model, solve_time):
# 返回的是True或者是False的结果.
    logger.info(allo_model)  # 打印一下模型分配的结果
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

            # 计算节点的深度
            node_depths = calculate_node_depths(edges)
            # 按照深度重新组织节点
            depths = reverseDict(node_depths)

            # 开始基于图进行推理
            heights = list(depths.keys())
            MAXHeight = max(heights)
            answerDict = {}  # 只有已经做过回答的subtask才会被放到这里面来
            # progress_bar = tqdm(total=len(steps))
            for i in range(MAXHeight):
                subtasks = depths[i]
                for subtaskid in subtasks:                
                    number = re.findall(r'\d+', subtaskid)
                    number = int(number[0]) if number else None
                    subtask = steps_dict[str(number)]
                    answer_MODEL = allo_model[number]
                    
                    # question 问题字符串
                    # 交待解决任务
                    sys_q = f"""There is a math_problem. I need you to solve it and give an answer.
Here is the problem:\n{question}

I have broken this math problem down into several smaller problems. I will assign you sub-problems one by one, and provide the results of the previous sub-problems as a reference for your reasoning.
Please solve the problem and respond according to mathematical logic.
    """  # 系统任务信息
                    
                    if len(answerDict)>0:
                        answersSoFar = f"""\nSo far, the answers to the resolved sub-problems are as follows: The format is Sub-problem-Id: xxx; Sub-problem: xxx; Answer: xxx."""
                        for key, value in answerDict.items():
                            answersSoFar += f"""\nSub-problem-Id: {key}; Sub-problem: {answerDict[key]['subtask']}; Answer: {answerDict[key]['answer']}."""
                        
                        predecessors = search_Predecessors(int_edges, number)
                        intersection = set(answerDict.keys()).intersection(set(predecessors))
                        count = len(intersection)
                        if count>0:
                            answersSoFar += f"""\nAmong them, sub-problems {predecessors} are directly related to this sub-problem, so please pay special attention to them."""
                    
                    subask = f"""\nThe sub-problem to solve now is xxx: {subtask}
Based on the information above, please provide a concise and clear answer"""
                    if len(answerDict)>0:
                        query = answersSoFar+subask
                    else:
                        query = subask
                    Q = [{'role':'system', 'content':sys_q},
                        {'role':'user', 'content':query},]                       
                    result = askLLM(clients, Q, tokens_path=tokens_path, model=answer_MODEL, temperature=1, max_tokens=300)
                    answerDict[number] = {'subtask':subtask, 'answer':result}
                    
            #         progress_bar.update(1)
            # progress_bar.close()

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
            # ifcorrect = askChatGPT(Q_judge, model=config["judgeCorrect_MODEL"], temperature=1)  # 要么是True, 要么是False
            ifcorrect = askLLM(clients, Q_judge, tokens_path=tokens_path, model=config['judgeCorrect_MODEL'], temperature=1, max_tokens=300)
            
            if 'True' in ifcorrect:
                # success_Q += 1
                logger.info('True')
                dict_str = str(allo_model)
                # 将字符串写入到指定的 txt 文件中
                with open(f'ModelAllocation/Cascade/Allo_search/{question_id}_{solve_time}.txt', 'a') as f:
                    f.write(dict_str+' True\n')
                
                num_allo_model = {k: model_mapping[v] for k, v in allo_model.items()}
                num_allo_model = num_allo_model.values()
                dict_str = str(num_allo_model)
                # 将字符串写入到指定的 txt 文件中
                with open(f'ModelAllocation/Cascade/Allo_search_num/{question_id}_{solve_time}.txt', 'a') as f:
                    f.write(dict_str+' True\n')
                return 'True'
            elif 'False' in ifcorrect:
                # unsuccess_Q += 1
                logger.info('False')
                dict_str = str(allo_model)
                # 将字符串写入到指定的 txt 文件中
                with open(f'ModelAllocation/Cascade/Allo_search/{question_id}_{solve_time}.txt', 'a') as f:
                    f.write(dict_str+' False\n')
                
                num_allo_model = {k: model_mapping[v] for k, v in allo_model.items()}
                num_allo_model = num_allo_model.values()
                dict_str = str(num_allo_model)
                # 将字符串写入到指定的 txt 文件中
                with open(f'ModelAllocation/Cascade/Allo_search_num/{question_id}_{solve_time}.txt', 'a') as f:
                    f.write(dict_str+' False\n')
                return 'False'

            success = True  # 任务未受中断,完整地结束了,所以标记为成功                
                
        except:
            attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
            logger.info(f"error: {attempts};  taskid: {question_id}")  # 生成过程出错了
                                    
    if attempts == MAX_TRY:
        logger.info(f'run error {MAX_TRY}+')    
        return 'error'


if __name__ == '__main__':    

    # 初始化token路径
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
    tokens_path = f'Tokens/token_usage_{formatted_now}.json'  # 这是记录token消耗的文件
    if not os.path.exists(tokens_path):
        with open(tokens_path, 'w') as f:
            json.dump({}, f)
    
    with open('CHAMP_config.json', 'r') as f:
        config = json.load(f)
    config['tokens_path'] = tokens_path
    
    logger, filename = setup_logger(aftername)
    # 数据存储路径
    file_path = '../Task_Datasets/CHAMP/all_champ_p.json'
    N = 270 # 在50个问题上进行测试.
    with open(file_path, 'r', encoding='utf-8') as file:
        problems = json.load(file)

    MAX_TRY = 2  # 错误尝试最大值  错误之后就尝试1次 一般重复一次解决不了的,重复N次也没有用
    question_ids = list(range(80, N))
    
    # with open('CHAMP_Decom_Steps_100.pkl', 'rb') as f:
    #     pre_steps = pickle.load(f)
    # print(pre_steps[0])
    # sys.exit(0)
    
    # 选择问题
    for question_id in tqdm(question_ids):
        
        with open(f'Decomposed_Steps/{question_id}.json', 'r', encoding='utf-8') as f:
            pre_steps = json.load(f)
        # 一个问题的分解文件中有若干个分解答案,所以需要区分
        
        question = problems[question_id]['problem_text']
        gold_answer = problems[question_id]['problem_answer']
            
        # 每个问题解2次        
        for solve_time in range(2):

            # 保存两个版本的分配结果.
            check_and_create_txt_file(f'ModelAllocation/Cascade/Allo_search/{question_id}_{solve_time}.txt')
            check_and_create_txt_file(f'ModelAllocation/Cascade/Allo_search_num/{question_id}_{solve_time}.txt')
                    
            logger.info('\n\n\n')
            logger.info(f'number id: {question_id}')
            
            # 加载预先load好的steps
            # 在这里进行模型分配的指定
            steps, steps_dict = pre_steps[solve_time]
            
            # 初始化最强的模型
            allo_model = {i + 1: 'gpt-4-turbo' for i in range(len(steps))}
            judgement = _reason(allo_model, solve_time=solve_time)          
            print(f'initial judgement: {judgement}')
            if judgement == 'True':
                # 如果原本可以做正确,就把大模型替换成较小的模型
                
                allo_model2 = {i + 1: 'gpt-4o-mini' for i in range(len(steps))}  # 一个中间状态,可以更快地降低模型
                judgement = _reason(allo_model2, solve_time=solve_time)
                if judgement == 'True':
                    allo_model = allo_model2
                    allo_model3 = {i + 1: 'llama3-70b' for i in range(len(steps))}  # 一个中间状态,可以更快地降低模型
                    judgement = _reason(allo_model3, solve_time=solve_time)
                    if judgement == 'True':
                        allo_model = allo_model3
                        allo_model4 = {i + 1: 'llama3-8b' for i in range(len(steps))}  # 一个中间状态,可以更快地降低模型
                        judgement = _reason(allo_model4, solve_time=solve_time)
                        if judgement == 'True':
                            allo_model = allo_model4
                    
                judgement = 'True'
                time_downgrading = 0
                while judgement == 'True' and time_downgrading < 15: 
                    allo_model = downgrading_vanilla(allo_model)
                    if allo_model == False:  # 已经没有可以降低的子任务了
                        break
                    judgement = _reason(allo_model, solve_time=solve_time)
                    time_downgrading += 1
                    # 如果有error的话,也被包含在其中了
            else:
                # 如果本来就错误,那就随机替换较小的模型.
                time_upgrading = 0
                while judgement == 'False' and time_upgrading < 3:
                    allo_model = downgrading_vanilla(allo_model)
                    judgement = _reason(allo_model, solve_time=solve_time)
                    time_upgrading += 1


    # 输出推理的token成本消耗
    with open(tokens_path, 'r') as f:
        token_usage = json.load(f)
        # logger.info(json.dumps(token_usage, indent=4))
        total_tokens, total_cost = CountCost(token_usage)
        # 打印结果
        logger.info(f"Total Tokens: {total_tokens}")
        logger.info(f"Total Cost: ${total_cost:.2f}")
    

