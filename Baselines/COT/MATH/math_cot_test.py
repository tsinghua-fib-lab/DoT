'''
需要把问题的prompts设计都与SCAN问题本身做适配
'''

# -*- coding: utf-8 -*-
import logging
import argparse
import copy
import json
import os
import pickle
import random
import re
import sys
from typing import List

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from MATH_cot_utils import *


sys.path.append(os.getcwd())

#from utils import *

#os.environ["http_proxy"] = "http://localhost:7890"
#os.environ["https_proxy"] = "http://localhost:7890"
setOpenAi(keyid = 1)

GPT_MODEL = "gpt-4-turbo"

def setup_logger():
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the global logging level to INFO
    
    # Define the format of log messages
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Log to a file
    filename = f'math_tot_{GPT_MODEL}_logfile.log'  # You may want to customize the filename
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Log to console (optional, for debugging purposes)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger, filename


if __name__ == '__main__':
    
    logger, filename = setup_logger()

    # 示例文件路径
    file_path = 'math200.json'
    X = 100
    count = 0
    questions = []
    types = []
    gold_answers = []
    # 打开文件并逐行读取内容
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    
    
    
    for item in dataset:
        questions.append(item['problem'])
        types.append(item['type'])
        gold_answers.append(remove_boxed(last_boxed_only_string(item['solution'])))
        count += 1
        if count >= X:
            break
    
    # question_ids = [29, 48]

    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    question_ids = list(range(100))
    #question_ids = [29, 48]
    
    
    # 选择问题
    for question_id in tqdm(question_ids):
        question = questions[question_id]
        type = types[question_id]
        gold_answer = gold_answers[question_id]
        
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('instruction content:')
        logger.info(question)


        attempts = 0
        success = False
        while attempts < 3 and not success:  # 如果遇到格式错误
            try:
                # 问题分解
                decompose_steps = decompose_sql(question, type,GPT_MODEL)
                # decompose_steps: "walk opposite right thrice after run opposite right" can be solved by: "run opposite right", "walk opposite right thrice".
                # print(decompose_steps)  # 基本没有问题
                
                # 分解后格式规范化
                steps, steps_dict = convert_steps_to_format(decompose_steps)
                # commands_decomposed: ['run opposite right', 'walk opposite right thrice']
                # print(steps_dict)
                num_steps = len(steps)
                print(steps)
                print(steps_dict)  # 只是加了一个问题的编号而已.
                print(len(steps))
                
                logger.info(f'Total number of steps: {num_steps}')
                # TOT 求解
                N = 2  # 每个子问题进行N次proposal
                M = 1  # 通过评估选出M个最好的proposal
                
                solution = []

                for i in range(num_steps):
                    subtask = steps[i]
                    sys_q = f"""There is a math_problem. I need you to solve it and give an answer.
                Here is the problem:\n{question}

                I have broken this math problem down into a series of smaller problems. I will assign you sub-problems one by one, and provide the results of the previous sub-problems as a reference for your reasoning.
                Please solve the problem and respond according to mathematical logic.
                """  

                    subask = f"""\nThe sub-problem to solve now is xxx: {subtask}
                Based on the information above, please provide a concise and clear answer"""
                    
                    if len(solution)==0:
                        # 第一个子问题
                        query = subask
                        Q = [{'role':'system', 'content':sys_q},
                            {'role':'user', 'content':query},]
                        for n in range(N):  # 一个子问题提问N次,获取N个解
                            result = askChatGPT(Q, model=GPT_MODEL, temperature=1)
                            eval_Q = Q + [{'role':'assistant', 'content':result}]
                            eval_Q = eval_Q + [{'role':'user', 'content':"Please provide a confidence rating for the accuracy of this solution, on a scale from 1 to 5. Only output the number."}]
                            score = askChatGPT(eval_Q, model=GPT_MODEL, temperature=1)
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
                                result = askChatGPT(Q, model=GPT_MODEL, temperature=1)
                                eval_Q = Q + [{'role':'assistant', 'content':result}]
                                eval_Q = eval_Q + [{'role':'user', 'content':"Please provide a confidence rating for the accuracy of this solution, on a scale from 1 to 5. Only output the number."}]
                                score = askChatGPT(eval_Q, model=GPT_MODEL, temperature=1)
                                score = int(score)
                                
                                temp_solution.append((solution[m][0]+score, solution[m][1]+[result]))  # 路径score累加
                        
                        # print(len(temp_solution))  # 此时temp_solution中应该有M*N种推理路径
                        solution = sorted(temp_solution, key=lambda x: x[0])
                        solution = solution[:M]  # 剪枝 M*N->M

                print(len(solution))
                printSeq(solution)
                
                # 从M个路径里挑一个最好的来问,也可以问完之后再评估一下选最好的答案

                user_q = f"""There is a math_problem:\n{question}

                I have broken this math problem down into a series of smaller problems and each sub-problem is solved.
                The sub-problems and their corresponding answers are as follows. (Format: Sub-problem-Id: xxx; Sub-problem: xxx; Answer: xxx.)"""

                for index, value in enumerate(solution[0][1]):  # 这里仅仅使用了最终得分最高的1条路径来总结得到final answer
                    user_q += f"""\nSub-problem-Id: {index}; Sub-problem: {steps[index]}; Answer: {value}."""

                Q.append({'role':'user', 'content':f"""Now that all the sub-problems have been solved, so what is the final answer?
                Please give the final answer without any additional explanation or clarification."""})
                finalResult = askChatGPT(Q, model=GPT_MODEL, temperature=1)
                print("THE FINAL RESULT IS: ", finalResult)
                # 让大语言模型来判断有没有回答正确
                judgeAnswer = {'role':'user', 'content':f"""Here is a math problem with a standard answer and a student's solution. Please help me determine if the student's solution is correct. If the numerical value are same, then it is correct.
                               
                Problem: {question}

                Standard answer: {gold_answer}

                Answer: {finalResult}

                If the student's answer is correct, just output True; otherwise, just output False.
                No explanation is required.
                """}

                Q_judge = [judgeAnswer]
                ifcorrect = askChatGPT(Q_judge, model=GPT_MODEL, temperature=1)  # 要么是True, 要么是False

                logger.info(f'final answer: {finalResult}')
                logger.info(f'gold answer: {gold_answer}')
                
                if 'True' in ifcorrect:
                    success_Q += 1
                    logger.info('correct')
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功
                elif 'False' in ifcorrect:
                    unsuccess_Q += 1
                    logger.info('incorrect')
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功 
                
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
    
    