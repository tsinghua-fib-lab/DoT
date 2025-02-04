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
openaiClient = setOpenAi(keyid = 3)
# llamaClient = Groq(  # 这个是Groq调用llama的api接口
#     api_key='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'
# )
llamaClient = OpenAI(
    api_key="EMPTY",
    base_url="http://101.6.69.60:8001/v1",
)
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "LLM_allocation_search-1004"



def _baseReason():
    heights = list(depths.keys())
    heights = [int(item) for item in heights]
    MAXHeight = max(heights)
    answerDict = {}  # 只有已经做过回答的subtask才会被放到这里面来
    for i in range(MAXHeight+1):
        subtasks = depths[str(i)]
        for subtaskid in subtasks:                
            number = re.findall(r'\d+', subtaskid)
            number = int(number[0]) if number else None
            subtask = steps_dict[str(number)]
            answer_MODEL = 'gpt-3.5-turbo'
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
            
            result, probs = askLLM_withprob(clients, Q, tokens_path=tokens_path, model=answer_MODEL, temperature=1, max_tokens=300)
            prob_values = list(probs.values())
            answerDict[number] = {'subtask':subtask, 'answer':result, 'probs':prob_values}
    return answerDict

    



def _reason(allo_model, solve_time):
    logger.info(allo_model)  # 打印一下模型分配的结果
    attempts = 0
    success = False
    while attempts < MAX_TRY and not success:  # 如果遇到格式错误
        try:            
            heights = list(depths.keys())
            heights = [int(item) for item in heights]
            MAXHeight = max(heights)
            answerDict = {}  # 只有已经做过回答的subtask才会被放到这里面来
            for i in range(MAXHeight+1):
                subtasks = depths[str(i)]
                for subtaskid in subtasks:                
                    number = re.findall(r'\d+', subtaskid)
                    number = int(number[0]) if number else None
                    subtask = steps_dict[str(number)]
                    # try:
                    answer_MODEL = allo_model[number]
                    # except:
                    #     print(subtasks)
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
                logger.info('True')
                dict_str = str(allo_model)
                # 将字符串写入到指定的 txt 文件中
                with open(f'ModelAllocation/Alpha-search/{question_id}_{solve_time}.txt', 'a') as f:
                    f.write(dict_str+' True\n')
                return 'True'
            elif 'False' in ifcorrect:
                logger.info('False')
                dict_str = str(allo_model)
                # 将字符串写入到指定的 txt 文件中
                with open(f'ModelAllocation/Alpha-search/{question_id}_{solve_time}.txt', 'a') as f:
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
    
    decom_ids = extract_numbers_from_filenames('Decomposed_Steps_1004')
    decom_ids.sort()

    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    MAX_TRY = 5
    DecomTime = 20
    
    # 选择问题
    for question_id in tqdm(decom_ids):
        if question_id==2:
            pass
        
        with open(f'Decomposed_Steps_1004/{question_id}.json', 'r', encoding='utf-8') as f:
            pre_steps = json.load(f)
            
        question = problems[question_id]['problem']
        type = problems[question_id]['type']
        gold_answer = problems[question_id]['solution']
            
        # 每个问题解2次        
        for solve_time in range(DecomTime):
        
            exi = check_and_create_txt_file(f'ModelAllocation/Alpha-search/{question_id}_{solve_time}.txt')
            if not exi:
                logger.info('\n\n\n')
                logger.info(f'number id: {question_id}  solve time: {solve_time}')

                steps, steps_dict, depths, int_edges = pre_steps[solve_time]
                depthlen = sum(len(value) for value in depths.values())
                # print(f"len(steps): {len(steps)}")
                # print(f"depthlen: {depthlen}")
                if len(steps) != depthlen:
                    depths = {str(i): [f'Step {i+1}'] for i in range(len(steps))}
                    int_edges = [[i, i+1] for i in range(1, len(steps))]
                
                subAnswerDict = _baseReason()
                alpha0 = []
                alpha04 = []
                alpha08 = []
                for i in range(1, len(steps)+1):
                    alpha0.append(quantile(subAnswerDict[i]['probs'], 0))  # frac = quantile(prob_values, alpha)
                    alpha04.append(quantile(subAnswerDict[i]['probs'], 0.4))
                    alpha08.append(quantile(subAnswerDict[i]['probs'], 0.8))

                sorted_list, sorted_indices = sort_with_indices(alpha04)  # 排序靠前的都是自信的回答,可都用小模型做
                allo_model = {sorted_indices[i]+1:'gpt-4-turbo' if sorted_list[i]<0.9 else 'llama3-8b' for i in range(len(sorted_indices))}
                cixu = [x + 1 for x in sorted_indices]
                judgement = _reason(allo_model, solve_time=solve_time)
                largeNum = sum(1 for x in alpha04 if x < 0.9) 
                smallNum = len(steps) - largeNum
                # 如果原本可以完成
                if judgement == 'True':
                    time_downgrading = 0
                    while judgement == 'True' and time_downgrading < largeNum: 
                        index = find_first_valid_key(cixu, allo_model)
                        allo_model[index] = 'llama3-8b'
                        judgement = _reason(allo_model, solve_time=solve_time)
                        time_downgrading += 1
                else:
                    time_upgrading = 0
                    while judgement == 'False' and time_upgrading < smallNum:
                        index = find_first_valid_key2(cixu, allo_model)
                        allo_model[index] = 'gpt-4-turbo'
                        judgement = _reason(allo_model, solve_time=solve_time)
                        time_upgrading += 1
                        
                # 有bug:
                # number id: 2  solve time: 0
                # {10: 'llama3-8b', 8: 'llama3-8b', 3: 'llama3-8b', 4: 'llama3-8b', 7: 'llama3-8b', 6: 'llama3-8b', 9: 'llama3-8b', 5: 'llama3-8b', 2: 'gpt-4-turbo', 1: 'gpt-4-turbo'}
                # False
                # {10: 'llama3-8b', 8: 'llama3-8b', 3: 'llama3-8b', 4: 'llama3-8b', 7: 'llama3-8b', 6: 'llama3-8b', 9: 'llama3-8b', 5: 'llama3-8b', 2: 'gpt-4-turbo', 1: 'gpt-4-turbo', None: 'gpt-4-turbo'}

    
    # 读取文件并打印结果以验证
    with open(tokens_path, 'r') as f:
        token_usage = json.load(f)
        # logger.info(json.dumps(token_usage, indent=4))
        total_tokens, total_cost = CountCost(token_usage)
        # 打印结果
        logger.info(f"Total Tokens: {total_tokens}")
        logger.info(f"Total Cost: ${total_cost:.2f}")
            
    
    