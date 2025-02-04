# -*- coding: utf-8 -*-
import argparse
import copy
import json
import logging
import os
import pickle
import random
import re
import sys
from collections import defaultdict, deque
from typing import Any, List

import ipydagred3
import networkx as nx
import numpy as np
import openai
from P3_TOT_utils import *
from tqdm import tqdm

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
client = setOpenAi(keyid = 0)

GPT_MODEL = "gpt-4o"



def setup_logger():
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the global logging level to INFO
    
    # Define the format of log messages
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Log to a file
    filename = f'P3_tot_{GPT_MODEL}.log'  # You may want to customize the filename
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
    puzzles = []


    with open("C:\\Users\Pluto\Desktop\TaDe\Task_Datasets\ProgramPuzzle\puzzles.json", "r") as f:
        puzzles = json.load(f)

    success_Q, unsuccess_Q, error_Q = 0, 0, 0
    N = 200
    
    
    # 选择问题
    for question_id in tqdm(range(N)):
        question = puzzles[question_id]["sat"]
        
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('instruction content:')
        logger.info(question)

        # 问题分解
        decompose_steps = decompose_sql(client, question, GPT_MODEL)
        steps, steps_dict = convert_steps_to_format(decompose_steps)
        num_steps = len(steps)
    
        logger.info(f'total number of steps: {num_steps}')

        # TOT 求解
        N = 2  # 每个子问题进行N次proposal
        M = 1  # 通过评估选出M个最好的proposal

        solution = []

        for i in range(num_steps):
            subtask = steps[i]
            sys_q = f"""
            There is a programming puzzle. Your final goal is to find an input for the function that will return True.
            
            Here is the problem:\n{question}

            Here, I have broken this puzzle down into a series of smaller problems. I will assign you sub-problems one by one, and provide the results of the previous sub-problems as a reference for your reasoning.
            
            Please solve the problem and respond according to the logic and results of the previous sub-problems.

        """  

            subask = f"""\nThe sub-steps to be solved now is:: {subtask}
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
                    answersSoFar = f"""\nSo far, the answers to the preceding sub-instructions are as follows: The format is Sub-problem-Id: xxx; Sub-problem: xxx; Answer: xxx."""
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
        #printSeq(solution)
        # 用额外的一次query再问一下最终的答案

        retry_count = 0
        success = False
        while retry_count < 3 and not success:
            try:
                user_q = f"""There is a natural language command representing a sequence of actions:\n{question}

                I have broken this action down into a series of smaller pieces and each sub-problem is solved.
                The sub-problems and their corresponding answers are as follows. """

                for index, value in enumerate(solution[0][1]):
                    user_q += f"""\nSub-problem-Id: {index}; Sub-problem: {steps[index]}; Answer: {value}."""

                user_q += "\n Given the results from the sub problems and their answer, so what is the final answer? (Try to concatenate some of the sub-problems  answers to get the final answer.)"

                final_Q = [{'role':'user', 'content':user_q}]

                s_result = askChatGPT(client, final_Q, model=GPT_MODEL, temperature=1)

                final_Q.append({'role':'assistant', 'content':s_result})
                final_Q.append({'role':'user', 'content':f"""Now that all the sub-tasks have been completed, so what is the correct input?
                Please only give the input in the format of a string and just give the answer without any additional explanation or clarification, no prefix or suffix.

                For example if the input should be x = 5, then you should only give the answer as 5 and not x = 5.
                For example, if the the input is list = [1,2,3], then you should only give the answer as [1,2,3] and not list = [1,2,3].
                """})
                finalResult = askChatGPT(client, final_Q, model=GPT_MODEL, temperature=1)
                
                logger.info(f'final answer: {finalResult}')
                logger.info('final answer type: {}'.format(puzzles[question_id]['ans_type']))
                
                exec(question)
                
                converted_result = convert_to_type(puzzles[question_id]['ans_type'], finalResult)
                
                result = sat(converted_result)
                
                if result == True:
                    success = True
                    success_Q += 1
                    logger.info('correct')
                else:
                    unsuccess_Q += 1
                    logger.info('incorrect')
                
                success = True
                
            except Exception as e:
                logger.error(f"An error occurred in final answer generation: {e}")
                retry_count += 1
                if retry_count < 3:
                    logger.info(f"Retrying final answer generation... ({retry_count}/3)")
                else:
                    logger.error("Maximum number of retries reached for final answer generation.")
                    error_Q += 1
                    logger.info('run error')
                    success = True

          
    logger.info(f'correct_Q: {success_Q}')
    logger.info(f'error_Q: {error_Q}')
    logger.info(f'incorrect_Q: {unsuccess_Q}')
    
    