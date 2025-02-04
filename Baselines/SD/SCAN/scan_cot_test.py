# -*- coding: utf-8 -*-
import argparse
import copy
import json
import os
import pickle
import random
import sys
import re
import numpy as np
import openai
from typing import List, Dict, Any
import ast
from scan_utils import *
from tqdm import tqdm

GPT_MODEL = "gpt-4-turbo-preview"  # [gpt-4-turbo-preview]
#GPT_MODEL = "gpt-3.5-turbo"  # [gpt-3.5-turbo-preview]

setOpenAi(keyid =4)



if __name__ == '__main__':
    
    logger, filename = setup_logger()

    # 示例文件路径
    tasks = []
    solutions = []
    count = 0
    N =100
    with open("/Users/natehu/Desktop/Tsinghua Research/TaDe/v1/scan_tasks.txt", 'r', encoding= 'utf-8') as file:
        for line in file:            
                question, actions = split_instruction(line.strip())            
                tasks.append(question)
                actions = [action.replace("I_", "") for action in actions.split()]
                solutions.append(actions)
                if count == N:
                    break

    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    question_ids = list(range(100))
    #question_ids = [33,13,25]
    
    
    # 选择问题
    for question_id in tqdm(question_ids):
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
                solution = run(question)
                actionList = solution.split()
                
                
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
          
    logger.info(f'correct_Q: {success_Q}')
    logger.info(f'error_Q: {error_Q}')
    logger.info(f'incorrect_Q: {unsuccess_Q}')
    
    