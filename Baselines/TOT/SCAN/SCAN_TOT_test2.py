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
from tqdm import tqdm

sys.path.append(os.getcwd())

from SCAN_TOT_utils import *

client = OpenAI(
    api_key="EMPTY",
    base_url="http://101.6.69.60:8001/v1",
)
GPT_MODEL = "llama3-8b-8192"

def setup_logger():
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the global logging level to INFO
    
    # Define the format of log messages
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Log to a file
    filename = f'SCAN_TOT_{GPT_MODEL}.log'  # You may want to customize the filename
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
    tasks = []
    solutions = []
    N = 50

    with open("C:\\Users\Pluto\Desktop\TaDe\Task_Datasets\SCAN\SCAN_all_tasks.txt", 'r', encoding='utf-8') as file:
        for line in file:
            if len(tasks) == N:
                break
            question, actions = split_instruction(line.strip())
            tasks.append(question)
            actions = [action.replace("I_", "") for action in actions.split()]
            solutions.append(actions)

    success_Q, unsuccess_Q, error_Q = 0, 0, 0
    
    
    # 选择问题
    for question_id in tqdm(range(len(tasks))):
        question = tasks[question_id]
        gold_answer = solutions[question_id]
        
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('instruction content:')
        logger.info(question)


        attempts = 0
        success = False
        while attempts < 3 and not success:  # 如果遇到格式错误
            try:
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

                    There is a natural language instruction representing a sequence of actions. I need you to translate this sentence from natural language into a standardized meta-action sequence."
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
                """  

                    subask = f"""\nThe sub-instruction to be converted now is:: {subtask}
                Based on the information above, please provide a concise and clear answer"""
                    
                    if len(solution)==0:
                        # 第一个子问题
                        query = subask
                        Q = [{'role':'system', 'content':sys_q},
                            {'role':'user', 'content':query},]
                        for n in range(N):  # 一个子问题提问N次,获取N个解
                            result = askChatGPT(client,Q, model=GPT_MODEL, temperature=1)
                            eval_Q = Q + [{'role':'assistant', 'content':result}]
                            eval_Q = eval_Q + [{'role':'user', 'content':"Please provide a confidence rating for the accuracy of this solution, on a scale from 1 to 5. Only output the number."}]
                            score = askChatGPT(client,eval_Q, model=GPT_MODEL, temperature=1)
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
                                    # print('warning')
                                    # print(index)
                                    # print(len(solution[m][1]))
                                    # print(len(steps))
                                    sys.exit(0)
                            query = answersSoFar+subask
                            Q = [{'role':'system', 'content':sys_q},
                                {'role':'user', 'content':query},]
                            for n in range(N):  # 一个子问题提问N次,获取N个解
                                result = askChatGPT(client,Q, model=GPT_MODEL, temperature=1)
                                eval_Q = Q + [{'role':'assistant', 'content':result}]
                                eval_Q = eval_Q + [{'role':'user', 'content':"Please provide a confidence rating for the accuracy of this solution, on a scale from 1 to 5. Only output the number."}]
                                score = askChatGPT(client,eval_Q, model=GPT_MODEL, temperature=1)
                                score = int(score)
                                
                                temp_solution.append((solution[m][0]+score, solution[m][1]+[result]))  # 路径score累加
                        
                        # print(len(temp_solution))  # 此时temp_solution中应该有M*N种推理路径
                        solution = sorted(temp_solution, key=lambda x: x[0])
                        solution = solution[:M]  # 剪枝 M*N->M

                # print(len(solution))
                # printSeq(solution)
            
                
                # 从M个路径里挑一个最好的来问,也可以问完之后再评估一下选最好的答案

                user_q = f"""There is a natural language command representing a sequence of actions:\n{question}

                I have broken this action down into a series of smaller pieces and each sub-problem is solved.
                The sub-problems and their corresponding answers are as follows. """

                for index, value in enumerate(solution[0][1]):  # 这里仅仅使用了最终得分最高的1条路径来总结得到final answer
                    user_q += f"""\nSub-problem-Id: {index}; Sub-problem: {steps[index]}; Answer: {value}."""

                user_q += "\n Given the results from the sub problems and their answer, so what is the final answer? (Try to concatenate some of the sub-problems  answers to get the final answer.)"

                final_Q = [{'role':'user', 'content':user_q}]
                
                finalResult = askChatGPT(client,final_Q, model=GPT_MODEL, temperature=1)
                actionSeq = sentenceRes2Actions(client, finalResult, GPT_MODEL)
                actionList = actionSeq.split()
                
                
                logger.info(f'final answer: {actionList}')
                logger.info(f'gold answer: {gold_answer}')
                
                
                if actionList == gold_answer:
                    success_Q += 1
                    print('correct')
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功
                    logger.info('correct')
                elif actionList != gold_answer:
                    unsuccess_Q += 1
                    print('incorrect')
                    logger.info('incorrect')
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功 
                
                success = True                      
               
            except Exception as e:
                attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
                print(f"error: {attempts};  taskid: {question_id};  Error_Message:  {e}")  # 生成过程出错了
        
        if attempts == 3:
            error_Q += 1
            logger.info('run error 3+')
        
        logger.info(f'accuracy: {success_Q/(success_Q+unsuccess_Q+error_Q)}')
    
    logger.info(f'correct_Q: {success_Q}')
    logger.info(f'error_Q: {error_Q}')
    logger.info(f'incorrect_Q: {unsuccess_Q}')
    
    