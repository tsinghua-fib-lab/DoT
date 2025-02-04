'''
MATH数学问题
'''
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
from MATH_Trys.MATH_utils import *
from utils import *

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
openaiClient = setOpenAi(keyid = 2)
llamaClient = Groq(  # 这个是Groq调用llama的api接口
    api_key='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'
)
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "LLM_allocation_search-0923"




def _reason(allo_model, solve_time):
    logger.info(allo_model)  # 打印一下模型分配的结果
    attempts = 0
    success = False
    while attempts < MAX_TRY and not success:  # 如果遇到格式错误
        try:            
            relations_test = construct_dependencies_without_traversal(clients, question, steps, config)  
            G1 = create_dag_from_string(relations_test)
            reduced_dependencies = list(G1.edges())
            edges = []
            for item in reduced_dependencies:
                edges.append((item[0][:item[0].find('[')].strip(), item[1][:item[1].find('[')].strip()))
            int_edges = [(int(e[0].split()[1]), int(e[1].split()[1])) for e in edges]
            node_depths = calculate_node_depths(edges)
            # 按照深度重新组织节点
            depths = reverseDict(node_depths)  # {0: ['Step 1'], 1: ['Step 2'], 2: ['Step 3']}
            heights = list(depths.keys())
            MAXHeight = max(heights)
            answerDict = {}  # 只有已经做过回答的subtask才会被放到这里面来
            for i in range(MAXHeight):
                subtasks = depths[i]
                for subtaskid in subtasks:                
                    number = re.findall(r'\d+', subtaskid)
                    number = int(number[0]) if number else None
                    subtask = steps_dict[str(number)]
                    answer_MODEL = allo_model[number]
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

            # 已经问完了所有的subtask,最后问一次得到最终的答案
            Q.append({'role':'assistant', 'content':result})
            Q.append({'role':'user', 'content':f"""Now that all the sub-problems have been solved, so what is the final answer?
Please give the final answer without any additional explanation or clarification."""})
            # finalResult = askChatGPT(Q, model=GPT_MODEL, temperature=1)
            finalResult = askLLM(clients, Q, tokens_path=tokens_path, model=config['finalSummarize_MODEL'], temperature=1, max_tokens=300)
            
            # logger.info('finalResult: ')
            # logger.info(finalResult)

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
                # success_Q += 1
                logger.info('correct')
                dict_str = str(allo_model)
                # 将字符串写入到指定的 txt 文件中
                with open(f'Allo_search/{question_id}_{solve_time}.txt', 'a') as f:
                    f.write(dict_str+' True\n')
                
                num_allo_model = {k: model_mapping[v] for k, v in allo_model.items()}
                num_allo_model = num_allo_model.values()
                dict_str = str(num_allo_model)
                # 将字符串写入到指定的 txt 文件中
                with open(f'Allo_search_num/{question_id}_{solve_time}.txt', 'a') as f:
                    f.write(dict_str+' True\n')
                return 'True'
            elif 'False' in ifcorrect:
                # unsuccess_Q += 1
                logger.info('error')
                dict_str = str(allo_model)
                # 将字符串写入到指定的 txt 文件中
                with open(f'Allo_search/{question_id}_{solve_time}.txt', 'a') as f:
                    f.write(dict_str+' False\n')
                
                num_allo_model = {k: model_mapping[v] for k, v in allo_model.items()}
                num_allo_model = num_allo_model.values()
                dict_str = str(num_allo_model)
                # 将字符串写入到指定的 txt 文件中
                with open(f'Allo_search_num/{question_id}_{solve_time}.txt', 'a') as f:
                    f.write(dict_str+' False\n')
                return 'False'
            
            success = True                   
            
        except:
            attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
            logger.info(f"error: {attempts};  taskid: {question_id}")  # 生成过程出错了
    
    if attempts == MAX_TRY:
        logger.info(f'run error {MAX_TRY}+')
        return 'error'
            
            
if __name__ == '__main__':
    start_time = time.time()
    # 初始化token路径
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
    tokens_path = f'Tokens/token_usage_{formatted_now}.json'  # 这是记录token消耗的文件
    if not os.path.exists(tokens_path):
        with open(tokens_path, 'w') as f:
            json.dump({}, f)
            
    logger, filename = setup_logger(aftername)
    
    with open('MATH_config.json', 'r') as f:
        config = json.load(f)
    config['tokens_path'] = tokens_path
        
    file_path = '../Task_Datasets/MATH/all_math_p.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        problems = json.load(file)

    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    N = 400
    MAX_TRY = 5  # 错误尝试最大值
    question_ids = list(range(0, N))
    
    # with open('MATH_Decom_Steps_100.pkl', 'rb') as f:
    #     pre_steps = pickle.load(f)
    
    # 选择问题
    for question_id in tqdm(question_ids):
        
        with open(f'Decomposed_Steps/{question_id}.json', 'r', encoding='utf-8') as f:
            pre_steps = json.load(f)
            
        question = problems[question_id]['problem']
        type = problems[question_id]['type']
        gold_answer = problems[question_id]['solution']
            
        # 每个问题解2次        
        for solve_time in range(2):
        
            # 保存两个版本的分配结果.
            check_and_create_txt_file(f'Allo_search/{question_id}_{solve_time}.txt')
            check_and_create_txt_file(f'Allo_search_num/{question_id}_{solve_time}.txt')
            
            logger.info('\n\n\n')
            logger.info(f'number id: {question_id}')

            # 加载预先load好的steps
            # 在这里进行模型分配的指定
            steps, steps_dict = pre_steps[solve_time]
            # allo_model = random_model_selection(len(steps))
            allo_model = allbest_allocation(len(steps))
            
            judgement = _reason(allo_model, solve_time=solve_time)
            print(f'initial judgement: {judgement}')
            if judgement == 'True':
                # 如果原本可以做正确,就把大模型替换成较小的模型.
                time_downgrading = 0
                while judgement == 'True' and time_downgrading < 25:
                    allo_model = downgrading_vanilla(allo_model)
                    judgement = _reason(allo_model, solve_time=solve_time)
                    time_downgrading += 1
            else:
                # 如果本来就错误,那就随机替换较小的模型.
                time_upgrading = 0
                while judgement == 'False' and time_upgrading < 3:
                    allo_model = upgrading(allo_model)
                    judgement = _reason(allo_model, solve_time=solve_time)
                    time_upgrading += 1
        


    end_time = time.time()
    # 计算运行时间
    elapsed_time = end_time - start_time
    hours, minutes, seconds = seconds_to_hms(elapsed_time)
    
    # 读取文件并打印结果以验证
    with open(tokens_path, 'r') as f:
        token_usage = json.load(f)
        # logger.info(json.dumps(token_usage, indent=4))
        total_tokens, total_cost = CountCost(token_usage)
        # 打印结果
        logger.info(f"Total Tokens: {total_tokens}")
        logger.info(f"Total Cost: ${total_cost:.2f}")
            
    
    