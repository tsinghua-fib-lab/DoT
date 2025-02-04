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
from p3_utils import *
from tqdm import tqdm

GPT_MODEL = "gpt-4-turbo-preview"  # [gpt-4-turbo-preview]
#GPT_MODEL = "gpt-3.5-turbo"  # [gpt-3.5-turbo-preview]

setOpenAi(keyid =4)



if __name__ == '__main__':
    
    logger, filename = setup_logger()

    # 示例文件路径
    puzzles = []


    with open("/Users/natehu/Desktop/Tsinghua Research/TaDe/dataset_gen_using_cot/p3_100_test_questions.json", "r") as f:
        puzzles = json.load(f)




    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    question_ids = list(range(100))
    
    
    
    # 选择问题
    for question_id in tqdm(question_ids):
        question = puzzles[question_id]["sat"]
        
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('instruction content:')
        logger.info(question)


        attempts = 0
        success = False
        while attempts < 3 and not success:  # 如果遇到格式错误
            try:
                # 问题分解
                solution = cot_solve(question)
                
                logger.info(f'final answer: {solution}')
                
                logger.info(f'final answer type: {puzzles[question_id]['ans_type']}')
                
                converted_result = convert_to_type(puzzles[question_id]['ans_type'], solution)
                exec(question)
                result = sat(converted_result)
                
                
                if result == True:
                    success_Q += 1
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功
                    logger.info('correct')
                else:
                    unsuccess_Q += 1
                    print('fail')
                    logger.info('fail')
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
    
    