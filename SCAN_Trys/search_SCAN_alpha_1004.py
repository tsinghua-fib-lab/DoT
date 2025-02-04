'''
基于alpha概率优化模型分配数据集构建
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
from SCAN_Trys.SCAN_utils import *
from utils import *

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
openaiClient = setOpenAi(keyid = 1)
# llamaClient = Groq(  # 这个是Groq调用llama的api接口
#     api_key='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'
# )
llamaClient = OpenAI(
    api_key="EMPTY",
    base_url="http://101.6.69.60:8001/v1",
)
# hxy: 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
# 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
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
            
            # question 问题字符串
            # 交待解决任务
            sys_q = f"""There is a natural language instruction representing a sequence of actions. I need you to translate this sentence from natural language into a standardized meta-action sequence."
Here is the instruction:\n{question}

I have broken this instruction down into some smaller instructions. I will assign you sub-instructions one by one, and provide the results of the previous sub-instructions as a reference for your reasoning.
Please organize your reasoning according to the combination and progression of actions.

For your reference, 13 examples for translation together with the corresponding explanations are as follows:

Q: "turn left"
A: "turn left" outputs "TURN LEFT".

Q: "turn right"
A: "turn right" outputs "TURN RIGHT".

Q: "jump left"
A: The output of “jump left” concatenates: the output of “turn left”, the output of “jump”. “turn left” outputs “TURN LEFT”. “jump” outputs “JUMP”. So concatenating the output of “turn left” and the output of “jump” leads to “TURN LEFT” + “JUMP”. So the output of “jump left” is “TURN LEFT” + “JUMP”.

Q: "run right"
A: The output of "run right" concatenates: the output of "turn right", the output of "run". "turn right" outputs "TURN RIGHT". "run" outputs "RUN". So concatenating the output of "turn right" and the output of "run" leads to "TURN RIGHT" + "RUN". So the output of "run right" is "TURN RIGHT" + "RUN".

Q: "look twice"
A: The output of "look twice" concatenates: the output of "look", the output of "look". "look" outputs "LOOK". So repeating the output of "look" two times leads to "LOOK" * 2. So the output of "look twice" is "LOOK" * 2.

Q: "run and look twice"
A: The output of "run and look twice" concate+nates: the output of "run", the output of "look twice". "run" outputs "RUN". "look twice" outputs "LOOK" * 2. So concatenating the output of "run" and the output of "look twice" leads to "RUN" + "LOOK" * 2. So the output of "run and look twice" is "RUN" + "LOOK" * 2.

Q: "jump right thrice"
A: The output of "jump right thrice" concatenates: the output of "jump right", the output of "jump right", the output of "jump right". "jump right" outputs "TURN RIGHT" + "JUMP". So repeating the output of "jump right" three times leads to ("TURN RIGHT" + "JUMP") * 3. So the output of "jump right thrice" is ("TURN RIGHT" + "JUMP") * 3.

Q: "walk after run"
A: The output of "walk after run" concatenates: the output of "run", the output of "walk". "run" outputs "RUN". "walk" outputs "WALK". So concatenating the output of "run" and the output of "walk" leads to "RUN" + "WALK". So the output of "walk after run" is "RUN" + "WALK".

Q: "turn opposite left"
A: The output of "turn opposite left" concatenates: the output of "turn left", the output of "turn left". "turn left" outputs "TURN LEFT". So repeating the output of "turn left" twice leads to "TURN LEFT" * 2. So the output of "turn opposite left" is "TURN LEFT" * 2.

Q: "turn around left"
A: The output of "turn around left" concatenates: the output of "turn left", the output of "turn left", the output of "turn left", the output of "turn left". "turn left" outputs "TURN LEFT". So repeating the output of "turn left" four times leads to "TURN LEFT" * 4. So the output of "turn around left" is "TURN LEFT" * 4. Q: "turn opposite right" A: The output of "turn opposite right" concatenates: the output of "turn right", the output of "turn right". "turn right" outputs "TURN RIGHT". So repeating the output of "turn right" twice leads to "TURN RIGHT" * 2. So the output of "turn opposite right" is "TURN RIGHT" * 2.

Q: "turn around right"
A: The output of "turn around right" concatenates: the output of "turn right", the output of "turn right", the output of "turn right", the output of "turn right". "turn right" outputs "TURN RIGHT". So repeating the output of "turn right" four times leads to "TURN RIGHT" * 4. So the output of "turn around right" is "TURN RIGHT" * 4.

Q: "walk opposite left"
A: The output of "walk opposite left" concatenates: the output of "turn opposite left", the output of "walk". "turn opposite left" outputs "TURN LEFT" * 2. "walk" outputs "WALK". So concatenating the output of "turn opposite left" and the output of "walk" leads to "TURN LEFT" * 2 + "WALK". So the output of "walk opposite left" is "TURN LEFT" * 2 + "WALK".

Q: "walk around left"
A: The output of "walk around left" concatenates: the output of "walk left", the output of "walk left", the output of "walk left", the output of "walk left". "walk left" outputs "TURN LEFT" + "WALK". So repeating the output of "walk around left" four times leads to ("TURN LEFT" + "WALK") * 4. So the output of "walk around left" is ("TURN LEFT" + "WALK") * 4.

Please pay attention to the use of parentheses.
"""  # 系统任务信息
            
            if len(answerDict)>0:
                answersSoFar = f"""\nSo far, the answers to the resolved sub-instructions are as follows: The format is Sub-instruction-Id: xxx; Sub-instruction: xxx; Answer: xxx."""
                for key, value in answerDict.items():
                    answersSoFar += f"""\nSub-instruction-Id: {key}; Sub-instruction: {answerDict[key]['subtask']}; Answer: {answerDict[key]['answer']}."""
                
                predecessors = search_Predecessors(int_edges, number)
                intersection = set(answerDict.keys()).intersection(set(predecessors))
                count = len(intersection)
                if count>0:
                    answersSoFar += f"""\nAmong them, sub-instructions {predecessors} are directly related to this sub-instruction, so please pay special attention to them."""
            
            
            subask = f"""\nThe sub-instruction to be converted now is:: {subtask}
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
            # 开始基于图进行推理
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
                    answer_MODEL = allo_model[number]
                    
                    # question 问题字符串
                    # 交待解决任务
                    sys_q = f"""There is a natural language instruction representing a sequence of actions. I need you to translate this sentence from natural language into a standardized meta-action sequence."
Here is the instruction:\n{question}

I have broken this instruction down into some smaller instructions. I will assign you sub-instructions one by one, and provide the results of the previous sub-instructions as a reference for your reasoning.
Please organize your reasoning according to the combination and progression of actions.

For your reference, 13 examples for translation together with the corresponding explanations are as follows:

Q: "turn left"
A: "turn left" outputs "TURN LEFT".

Q: "turn right"
A: "turn right" outputs "TURN RIGHT".

Q: "jump left"
A: The output of “jump left” concatenates: the output of “turn left”, the output of “jump”. “turn left” outputs “TURN LEFT”. “jump” outputs “JUMP”. So concatenating the output of “turn left” and the output of “jump” leads to “TURN LEFT” + “JUMP”. So the output of “jump left” is “TURN LEFT” + “JUMP”.

Q: "run right"
A: The output of "run right" concatenates: the output of "turn right", the output of "run". "turn right" outputs "TURN RIGHT". "run" outputs "RUN". So concatenating the output of "turn right" and the output of "run" leads to "TURN RIGHT" + "RUN". So the output of "run right" is "TURN RIGHT" + "RUN".

Q: "look twice"
A: The output of "look twice" concatenates: the output of "look", the output of "look". "look" outputs "LOOK". So repeating the output of "look" two times leads to "LOOK" * 2. So the output of "look twice" is "LOOK" * 2.

Q: "run and look twice"
A: The output of "run and look twice" concate+nates: the output of "run", the output of "look twice". "run" outputs "RUN". "look twice" outputs "LOOK" * 2. So concatenating the output of "run" and the output of "look twice" leads to "RUN" + "LOOK" * 2. So the output of "run and look twice" is "RUN" + "LOOK" * 2.

Q: "jump right thrice"
A: The output of "jump right thrice" concatenates: the output of "jump right", the output of "jump right", the output of "jump right". "jump right" outputs "TURN RIGHT" + "JUMP". So repeating the output of "jump right" three times leads to ("TURN RIGHT" + "JUMP") * 3. So the output of "jump right thrice" is ("TURN RIGHT" + "JUMP") * 3.

Q: "walk after run"
A: The output of "walk after run" concatenates: the output of "run", the output of "walk". "run" outputs "RUN". "walk" outputs "WALK". So concatenating the output of "run" and the output of "walk" leads to "RUN" + "WALK". So the output of "walk after run" is "RUN" + "WALK".

Q: "turn opposite left"
A: The output of "turn opposite left" concatenates: the output of "turn left", the output of "turn left". "turn left" outputs "TURN LEFT". So repeating the output of "turn left" twice leads to "TURN LEFT" * 2. So the output of "turn opposite left" is "TURN LEFT" * 2.

Q: "turn around left"
A: The output of "turn around left" concatenates: the output of "turn left", the output of "turn left", the output of "turn left", the output of "turn left". "turn left" outputs "TURN LEFT". So repeating the output of "turn left" four times leads to "TURN LEFT" * 4. So the output of "turn around left" is "TURN LEFT" * 4. Q: "turn opposite right" A: The output of "turn opposite right" concatenates: the output of "turn right", the output of "turn right". "turn right" outputs "TURN RIGHT". So repeating the output of "turn right" twice leads to "TURN RIGHT" * 2. So the output of "turn opposite right" is "TURN RIGHT" * 2.

Q: "turn around right"
A: The output of "turn around right" concatenates: the output of "turn right", the output of "turn right", the output of "turn right", the output of "turn right". "turn right" outputs "TURN RIGHT". So repeating the output of "turn right" four times leads to "TURN RIGHT" * 4. So the output of "turn around right" is "TURN RIGHT" * 4.

Q: "walk opposite left"
A: The output of "walk opposite left" concatenates: the output of "turn opposite left", the output of "walk". "turn opposite left" outputs "TURN LEFT" * 2. "walk" outputs "WALK". So concatenating the output of "turn opposite left" and the output of "walk" leads to "TURN LEFT" * 2 + "WALK". So the output of "walk opposite left" is "TURN LEFT" * 2 + "WALK".

Q: "walk around left"
A: The output of "walk around left" concatenates: the output of "walk left", the output of "walk left", the output of "walk left", the output of "walk left". "walk left" outputs "TURN LEFT" + "WALK". So repeating the output of "walk around left" four times leads to ("TURN LEFT" + "WALK") * 4. So the output of "walk around left" is ("TURN LEFT" + "WALK") * 4.

Please pay attention to the use of parentheses.
"""  # 系统任务信息
                    
                    if len(answerDict)>0:
                        answersSoFar = f"""\nSo far, the answers to the resolved sub-instructions are as follows: The format is Sub-instruction-Id: xxx; Sub-instruction: xxx; Answer: xxx."""
                        for key, value in answerDict.items():
                            answersSoFar += f"""\nSub-instruction-Id: {key}; Sub-instruction: {answerDict[key]['subtask']}; Answer: {answerDict[key]['answer']}."""
                        
                        predecessors = search_Predecessors(int_edges, number)
                        intersection = set(answerDict.keys()).intersection(set(predecessors))
                        count = len(intersection)
                        if count>0:
                            answersSoFar += f"""\nAmong them, sub-instructions {predecessors} are directly related to this sub-instruction, so please pay special attention to them."""
                    
                    
                    subask = f"""\nThe sub-instruction to be converted now is:: {subtask}
Based on the information above, please provide a concise and clear answer"""

                    if len(answerDict)>0:
                        query = answersSoFar+subask
                    else:
                        query = subask

                    Q = [{'role':'system', 'content':sys_q},
                        {'role':'user', 'content':query},]    
                    result = askLLM(clients, Q, tokens_path=tokens_path, model=answer_MODEL, temperature=1, max_tokens=300)
                    answerDict[number] = {'subtask':subtask, 'answer':result}

            Q.append({'role':'assistant', 'content':result})
            Q.append({'role':'user', 'content':f"""Now that all the sub-instructions have been completed, so what is the final answer?
Please give the final action sequence without any additional explanation or clarification."""})
            # finalResult = askChatGPT(Q, model=config["finalSummarize_MODEL"], temperature=1)
            finalResult = askLLM(clients, Q, tokens_path=tokens_path, model=config['finalSummarize_MODEL'], temperature=1)

            actionSeq = sentenceRes2Actions(clients, finalResult, config)
            actionList = actionSeq.split()
            ifcorrect = actionList == solutions[question_id]
            if ifcorrect:
                # success_Q += 1
                logger.info('True')
                dict_str = str(allo_model)
                # 将字符串写入到指定的 txt 文件中
                with open(f'ModelAllocation/Alpha-search/{question_id}_{solve_time}.txt', 'a') as f:
                    f.write(dict_str+' True\n')
                success = True  # 任务未受中断,完整地结束了,所以标记为成功           
                return 'True'
            else:
                # unsuccess_Q += 1
                logger.info('False')
                dict_str = str(allo_model)
                # 将字符串写入到指定的 txt 文件中
                with open(f'ModelAllocation/Alpha-search/{question_id}_{solve_time}.txt', 'a') as f:
                    f.write(dict_str+' False\n')
                success = True  # 任务未受中断,完整地结束了,所以标记为成功                    
                return 'False'
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
    
    with open('SCAN_config.json', 'r') as f:
        config = json.load(f)
    config['tokens_path'] = tokens_path
    
    logger, filename = setup_logger(aftername)
    # 数据存储路径
    file_path = '../Task_Datasets/SCAN/SCAN_all_tasks.txt'
    tasks = []
    solutions = []
    N = 100
    count = 0
    # 打开文件并逐行读取内容
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:            
            question, actions = extract_in_out(line.strip())            
            tasks.append(question)
            actions = [action.replace("I_", "") for action in actions.split()]
            solutions.append(actions)
            if count == N:
                break
    

    MAX_TRY = 5  # 错误尝试最大值    
    decom_ids = extract_numbers_from_filenames('Decomposed_Steps_1004')
    decom_ids.sort()
    
    for question_id in tqdm(decom_ids):  # 在分解好的问题上进行搜索
        
        with open(f'Decomposed_Steps_1004/{question_id}.json', 'r', encoding='utf-8') as f:
            pre_steps = json.load(f)
        
        question = tasks[question_id]
            
        # 每个问题解20次        
        for solve_time in range(20):

            # 保存两个版本的分配结果.
            exi = check_and_create_txt_file(f'ModelAllocation/Alpha-search/{question_id}_{solve_time}.txt')
            if not exi:
                logger.info('\n\n\n')
                logger.info(f'number id: {question_id}  solve time: {solve_time}')
                try:
                    steps, steps_dict, depths, int_edges = pre_steps[solve_time]
                except:
                    print('continue!')
                    continue
                depthlen = sum(len(value) for value in depths.values())
                if len(steps) != depthlen:
                    depths = {str(i): [f'Step {i+1}'] for i in range(len(steps))}
                    int_edges = [[i, i+1] for i in range(1, len(steps))]
                
                subAnswerDict = _baseReason()  # 这一步是为了获取所有子问题试探性回答的概率值
                alpha0 = []
                alpha04 = []
                alpha08 = []
                for i in range(1, len(steps)+1):
                    alpha0.append(quantile(subAnswerDict[i]['probs'], 0))  # frac = quantile(prob_values, alpha)
                    alpha04.append(quantile(subAnswerDict[i]['probs'], 0.4))
                    alpha08.append(quantile(subAnswerDict[i]['probs'], 0.8))
                # print(f"alpha04: {alpha04}")  # 选择看这个
                # print(alpha04)
                sorted_list, sorted_indices = sort_with_indices(alpha04)  # 排序靠前的都是自信的回答,可都用小模型做

                allo_model = {sorted_indices[i]+1:'gpt-4-turbo' if sorted_list[i]<0.9 else 'llama3-8b' for i in range(len(sorted_indices))}

                cixu = [x + 1 for x in sorted_indices]
                judgement = _reason(allo_model, solve_time=solve_time)
                largeNum = sum(1 for x in alpha04 if x < 0.9)  # 大模型的数量
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
                    # 如果本来就错误,那就替换更大的模型
                    time_upgrading = 0
                    while judgement == 'False' and time_upgrading < smallNum:
                        index = find_first_valid_key2(cixu, allo_model)
                        allo_model[index] = 'gpt-4-turbo'
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
    

