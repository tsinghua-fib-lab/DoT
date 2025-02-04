'''
LLM自动化评估任务难度以及分配LLM进行解决
TODO 怎么把推理进度条给隐藏了
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
import logging

from puzzle_utils import *

from utils import *

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
openaiClient = setOpenAi(keyid = 0)
llamaClient = OpenAI(
    api_key="EMPTY",
    base_url="http://101.6.69.60:8000/v1",
)
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "最终方案测试 Step2 纯4o"
MAX_TRY = 3

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
    
    with open("puzzles.json", "r") as f:
        puzzles = json.load(f)
        
    with open('Puzzle_config.json', 'r') as f:
        config = json.load(f)
    config['tokens_path'] = tokens_path
        
#     question_ids = [29, 48, 61, 81, 84, 108, 110, 114, 115, 188, 192, 195, 212, 226, 257, 277, 283, 285, 295, 320, 339, 341, 343, 359, 393, 416, 434, 449, 452, 457, 462, 495, 500, 519, 533, 546, 554, 555, 556, 576, 582, 606, 637, 641, 653, 675, 723, 752, 759, 811, 
# 820, 822, 827, 828, 853, 876, 881, 897, 920, 929, 937, 951, 1021, 1022, 1037, 1069, 1071, 1110, 1123, 1128, 1130, 1145, 1169, 1201, 1227, 1233, 1234, 1235, 1240, 1255, 1265, 1301, 1316, 1350, 1358, 1373, 1420, 1440, 1454, 1512, 1529, 1533, 1596, 
# 1597, 1607, 1627, 1658, 1660, 1697, 1707]


    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    # 选择问题
    N = 100
    
    f = open('TmpRes/step2In_Puzzle_last.json', 'r')
    content = f.read()
    middleRes = json.loads(content) 
    
    
    for question_id in tqdm(range(N)):
        question = puzzles[question_id]['sat']
        
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('label id: '+puzzles[question_id]['name'])
        logger.info('puzzle content:')
        logger.info(question)
        
        attempts = 0
        success = False
        while attempts < MAX_TRY and not success:  # 如果遇到格式错误
            try:
                steps, steps_dict, allo_model, depths, int_edges = middleRes[str(question_id)]['steps'], middleRes[str(question_id)]['steps_dict'], middleRes[str(question_id)]['allo_model'], middleRes[str(question_id)]['depths'], middleRes[str(question_id)]['int_edges']
                depths = {int(k): v for k, v in depths.items()}
                heights = list(depths.keys())
                MAXHeight = max(heights)
                answerDict = {}  # 只有已经做过回答的subtask才会被放到这里面来
                
                for i in range(MAXHeight):
                    subtasks = depths[i]
                    for subtaskid in subtasks:

                        number = re.findall(r'\d+', subtaskid)
                        number = int(number[0]) if number else None
                        subtask = steps_dict[str(number)]
                        # answer_MODEL = allo_model[number-1]
                        answer_MODEL = 'gpt-4o'
                        
                        # question 问题字符串
                        # 交待解决任务
                        sys_q = f"""You will be provided with a Programming Puzzle. Your task is to find an input that will make the program return True.
Here is the puzzle:\n{question}

The data type of your final answer should be {puzzles[question_id]['ans_type']}.
I have broken this puzzle down into many easier subtasks. I will assign you sub-tasks one by one, and provide the results of the previous sub-tasks as a reference for your reasoning.
Please follow the logical sequence of our subtasks to find the correct input."""
                        
                        if len(answerDict)>0:
                            answersSoFar = f"""\nSo far, the answers to the resolved sub-tasks are as follows: The format is SubtaskId: xxx; Subtask: xxx; Answer: xxx."""
                            for key, value in answerDict.items():
                                answersSoFar += f"""\nSubtaskId: {key}; Subtask: {answerDict[key]['subtask']}; Answer: {answerDict[key]['answer']}."""
                            
                            predecessors = search_Predecessors(int_edges, number)
                            intersection = set(answerDict.keys()).intersection(set(predecessors))
                            count = len(intersection)
                            if count>0:
                                answersSoFar += f"""\nAmong them, sub-tasks {predecessors} are directly related to this sub-task, so please pay special attention to them."""
                        
                        
                        subask = f"""\nNow the subtask is: {subtask}
Based on the information above, please provide a concise and clear answer to this sub-task in one or two sentences.."""

                        if len(answerDict)>0:
                            query = answersSoFar+subask
                        else:
                            query = subask

                        Q = [{'role':'system', 'content':sys_q},
                            {'role':'user', 'content':query},]
                            
                        # print(subtaskid)
                        # print(subtask)
                        # print('**********Question**********')
                        # print(Q)
                        # result = askChatGPT(Q, model='gpt-3.5-turbo', temperature=1)
                        result = askLLM(clients, Q, tokens_path=tokens_path, model=answer_MODEL, temperature=1, max_tokens=300)
                        # print('Answer:', result)
                        # print('\n\n\n')
                        answerDict[number] = {'subtask':subtask, 'answer':result}

                # 已经问完了所有的subtask,最后问一次得到最终的答案
                Q.append({'role':'assistant', 'content':result})
                Q.append({'role':'user', 'content':f"""Now that all the sub-tasks have been completed, so what is the correct input?
Please give the input in the format of a string and just give the answer without any additional explanation or clarification."""})
                # finalResult = askChatGPT(Q, model=GPT_MODEL, temperature=1)
                finalResult = askLLM(clients, Q, tokens_path=tokens_path, model=config['finalSummarize_MODEL'], temperature=1)
                # print('图上推理 done')

                finalResult = remove_quotes(finalResult)

                exec(question)
                # 根据ans_type_str将结果转换为相应的类型
                # print('final Result')
                # print(finalResult)
                # print('should answer type')
                # print(puzzles[question_id]['ans_type'])
                converted_result = convert_to_type(puzzles[question_id]['ans_type'], finalResult)
                # print('converted_result:\n')
                # print(converted_result)
                # print('converted_result type') 
                # print(type(converted_result))
                
                result = sat(converted_result)
                # print("函数运行结果为：", result)
                if result == True:
                    success_Q += 1
                    # print('success')
                    logger.info('True->Success')
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功
                else:
                    unsuccess_Q += 1
                    # print('fail')
                    logger.info('False->Fail')
                    attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
                
            
            except:
                attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
                print(f"error: {attempts};  taskid: {question_id}")  # 生成过程出错了
                
        if attempts == MAX_TRY:
            error_Q += 1
            logger.info('run error 5+')
            
    end_time = time.time()
    # 计算运行时间
    elapsed_time = end_time - start_time
    hours, minutes, seconds = seconds_to_hms(elapsed_time)
    logger.info(f"100 solving 运行耗时: {hours}h, {minutes}min, {seconds}s")   
    
    logger.info(f'\n{tokens_path}')
    logger.info(f'Correct_Q: {success_Q}')
    logger.info(f'False_Q: {unsuccess_Q}')
    logger.info(f'Error_Q: {error_Q}')  
    
    # 读取文件并打印结果以验证
    with open(tokens_path, 'r') as f:
        token_usage = json.load(f)
        # logger.info(json.dumps(token_usage, indent=4))
        total_tokens, total_cost = CountCost(token_usage)
        # 打印结果
        logger.info(f"Total Tokens: {total_tokens}")
        logger.info(f"Total Cost: ${total_cost:.2f}")
    
    
    