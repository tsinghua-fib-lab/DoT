'''
需要把问题的prompts设计都与SCAN问题本身做适配
'''

import argparse
import copy
import json
# -*- coding: utf-8 -*-
import logging
import os
import pickle
import random
import re
import sys
from typing import List

import numpy as np
import openai
from csqa_utils import *
from openai import OpenAI
from tqdm import tqdm

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

client = OpenAI(
    api_key="EMPTY",
    base_url="http://101.6.69.60:8000/v1",
)

GPT_MODEL = "llama3-8b-8192"



if __name__ == '__main__':
    
    logger, filename = setup_logger(GPT_MODEL)

    
    dataset = []
    with open('C:\\Users\Pluto\Desktop\TaDe\Task_Datasets\CSQA\\train_rand_split.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    N = 200
    dataset = dataset[:N]
    
    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    question_ids = range(len(dataset))

    
    # 选择问题
    for question_id in tqdm(question_ids):
        question = dataset[question_id]["question"]["stem"]
        choices = dataset[question_id]["question"]["choices"]
        gold_answer = dataset[question_id]["answerKey"]

        
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('instruction content:')
        logger.info(question)
        logger.info(choices)


        attempts = 0
        success = False
        while attempts < 3 and not success:  # 如果遇到格式错误
            try:
                # 问题分解
                decompose_steps = decompose_sql(client, question, choices, GPT_MODEL)
                steps, steps_dict = convert_steps_to_format(decompose_steps)
                num_steps = len(steps)
                print(steps)
                print(steps_dict)  # 只是加了一个问题的编号而已.
                # TOT 求解
                N = 3  # 每个子问题进行N次proposal
                M = 1  # 通过评估选出M个最好的proposal

                solution = []

                for i in range(num_steps):
                    subtask = steps[i]
                    sys_q = f"""
                    There is a common sense question. I need you to solve it and give an answer.
                    
                    Here is the problem:\n{question}
                    
                    The choices are: {choices}

                I have broken this  problem down into a series of smaller problems. I will assign you sub-problems one by one, and provide the results of the previous sub-problems as a reference for your reasoning.
                Please solve the problem and respond according to logic.
                """  

                    subask = f"""\nThe sub-problem to solve now is xxx: {subtask}
                Based on the information above, please provide a concise and clear answer"""
                    
                    if len(solution)==0:
                        # 第一个子问题
                        query = subask
                        Q = [{'role':'system', 'content':sys_q},
                            {'role':'user', 'content':query},]
                        for n in range(N):  # 一个子问题提问N次,获取N个解
                            result = askChatGPT(client, Q, model=GPT_MODEL, temperature=1)
                            eval_Q = Q + [{'role':'assistant', 'content':result}]
                            eval_Q = eval_Q + [{'role':'user', 'content':"Please provide a confidence rating for the accuracy of this solution, on a scale from 1 to 5. Only output the number."}]
                            score = askChatGPT(client, eval_Q, model=GPT_MODEL, temperature=1)
                            score = int(score)
                            
                            solution.append((score, [result]))  # 维护一整条推理路径
                            
                        solution = sorted(solution, key=lambda x: x[0])
                        solution = solution[:M]  # 剪枝
                    else:
                        temp_solution = []
                        for m in range(M):  # 因为剪枝动态维护M个推理路径
                            answersSoFar = f"""\nSo far, the answers to the preceding sub-problems are as follows: The format is Sub-problem-Id: xxx; Sub-problem: xxx; Answer: xxx."""
                            for index, value in enumerate(solution[m][1]):
                                try:
                                    answersSoFar += f"""\nSub-problem-Id: {index}; Sub-problem: {steps[index]}; Answer: {value}."""
                                except:
                                    print('warning')
                                    print(index)
                                    print(len(solution[m][1]))
                                    print(len(steps))
                                    sys.exit(0)
                            query = answersSoFar+subask
                            Q = [{'role':'system', 'content':sys_q},
                                {'role':'user', 'content':query},]
                            for n in range(N):  # 一个子问题提问N次,获取N个解
                                result = askChatGPT(client, Q, model=GPT_MODEL, temperature=1)
                                eval_Q = Q + [{'role':'assistant', 'content':result}]
                                eval_Q = eval_Q + [{'role':'user', 'content':"Please provide a confidence rating for the accuracy of this solution, on a scale from 1 to 5. Only output the number."}]
                                score = askChatGPT(client, eval_Q, model=GPT_MODEL, temperature=1)
                                score = int(score)
                                
                                temp_solution.append((solution[m][0]+score, solution[m][1]+[result]))  # 路径score累加
                        
                        # print(len(temp_solution))  # 此时temp_solution中应该有M*N种推理路径
                        solution = sorted(temp_solution, key=lambda x: x[0])
                        solution = solution[:M]  # 剪枝 M*N->M

                print(len(solution))
                printSeq(solution)
                

                # 从M个路径里挑一个最好的来问,也可以问完之后再评估一下选最好的答案

                user_q = f"""There is a common sense question:\n{question}

                I have broken this math problem down into a series of smaller problems and each sub-problem is solved.
                The sub-problems and their corresponding answers are as follows. (Format: Sub-problem-Id: xxx; Sub-problem: xxx; Answer: xxx.)"""

                for index, value in enumerate(solution[0][1]):  # 这里仅仅使用了最终得分最高的1条路径来总结得到final answer
                    user_q += f"""\nSub-problem-Id: {index}; Sub-problem: {steps[index]}; Answer: {value}."""

                Q.append({'role':'user', 'content':f"""Now that all the sub-problems have been solved, so what is the final answer?
                Please Onle give the letter of the correct answer, no need to provide the reason or the process or information. 
                For example: If the answer is A, please output 'A'. """})
                finalResult = askChatGPT(client, Q, model=GPT_MODEL, temperature=1)

                
                logger.info(f'final answer: {finalResult}')
                logger.info(f'gold answer: {gold_answer}')

                if finalResult == gold_answer:
                    print('correct')
                    success_Q += 1
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功
                    logger.info('correct')
                elif finalResult != gold_answer:
                    unsuccess_Q += 1
                    print('error')
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功          
                    logger.info('error')             
                
                
                
                success = True                      
               
            except Exception as e:
                attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
                print(f"error: {attempts};  taskid: {question_id};  Error_Message:  {e}")  # 生成过程出错了
        
        if attempts == 3:
            error_Q += 1
            logger.info('run error 3+')
          
    logger.info(f'correct_Q: {success_Q}')
    logger.info(f'error_Q: {error_Q}')
    logger.info(f'incorrect_Q: {unsuccess_Q}')
    logger.info(f'sum_Q: {success_Q+error_Q+unsuccess_Q}')
    
    