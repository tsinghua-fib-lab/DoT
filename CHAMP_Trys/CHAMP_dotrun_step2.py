'''
MATH数学问题
在least2most里没有设计好的prompts了
自己寻找适配CHAMP数据集的prompts设计
使用大语言模型执行子任务难度评估与模型分配 -> 目标方案, 最终方案
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
from tqdm import tqdm
sys.path.append('../')
from CHAMP_Trys.CHAMP_utils import *
from utils import *

# client定义需要满足如下调用方式: client.chat.completions.create(model,messages = messages), 详见askLLM函数
openaiClient = setOpenAi(keyid = 0)
llamaClient = setLocal()
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "final_version-step2"

if __name__ == '__main__':
    
    start_time = time.time()
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
    N = 200 # 在50个问题上进行测试.
    with open(file_path, 'r', encoding='utf-8') as file:
        problems = json.load(file)
    

    f = open('TmpRes/step2In_champ_last.json', 'r')
    content = f.read()
    middleRes = json.loads(content) 
    
    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    MAX_TRY = 8  # 错误尝试最大值
    question_ids = list(range(N))
    
    # 选择问题
    for question_id in tqdm(question_ids):
                
        question = problems[question_id]['problem_text']
        gold_answer = problems[question_id]['problem_answer']
        
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('instruction content:')
        logger.info(question)

        attempts = 0
        success = False
        while attempts < MAX_TRY and not success:  # 如果遇到格式错误
            try:
                steps, steps_dict, allo_model, depths, int_edges = middleRes[str(question_id)]['steps'], middleRes[str(question_id)]['steps_dict'], middleRes[str(question_id)]['allo_model'], middleRes[str(question_id)]['depths'], middleRes[str(question_id)]['int_edges']
                # 这些量都是需要加载的
                depths = {int(k): v for k, v in depths.items()}

                
                # 开始基于图进行推理
                heights = list(depths.keys())
                MAXHeight = max(heights)
                answerDict = {}
                progress_bar = tqdm(total=len(steps))
                for i in range(MAXHeight):
                    subtasks = depths[i]
                    for subtaskid in subtasks:                
                        number = re.findall(r'\d+', subtaskid)
                        number = int(number[0]) if number else None
                        subtask = steps_dict[str(number)]
                        answer_MODEL = allo_model[number-1]
                        
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
                        progress_bar.update(1)

                progress_bar.close()

                # 已经问完了所有的subtask,最后问一次得到最终的答案
                Q.append({'role':'assistant', 'content':result})
                Q.append({'role':'user', 'content':f"""Now that all the sub-problems have been solved, so what is the final answer?
Please give the final answer without any additional explanation or clarification."""})
                # finalResult = askChatGPT(Q, model=config["finalSummarize_MODEL"], temperature=1)
                finalResult = askLLM(clients, Q, tokens_path=tokens_path, model=config['finalSummarize_MODEL'], temperature=1, max_tokens=300)
                
                print('图上推理 done')
                logger.info('finalResult:\n')
                logger.info(finalResult)
                
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
                    success_Q += 1
                    logger.info('correct')
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功
                elif 'False' in ifcorrect:
                    unsuccess_Q += 1
                    logger.info('error')
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功                       
                
            except:
                attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
                logger.info(f"error: {attempts};  taskid: {question_id}")  # 生成过程出错了
        
        if attempts == MAX_TRY:
            error_Q += 1
            logger.info(f'run error {MAX_TRY}+')
            
    end_time = time.time()
    # 计算运行时间
    elapsed_time = end_time - start_time
    # print(f"程序运行耗时: {elapsed_time} 秒")
    hours, minutes, seconds = seconds_to_hms(elapsed_time)
    logger.info(f"100 solving 运行耗时: {hours}h, {minutes}min, {seconds}s")
    logger.info(f'\n{tokens_path}')
    logger.info(f'correct_Q: {success_Q}')
    logger.info(f'error_Q: {error_Q}')

    # 读取文件并打印结果以验证
    with open(tokens_path, 'r') as f:
        token_usage = json.load(f)
        # logger.info(json.dumps(token_usage, indent=4))
        total_tokens, total_cost = CountCost(token_usage)
        # 打印结果
        logger.info(f"Total Tokens: {total_tokens}")
        logger.info(f"Total Cost: ${total_cost:.2f}")
    

