'''
MATH数学问题
在least2most里没有设计好的prompts了
自己寻找适配CHAMP数据集的prompts设计
使用大语言模型执行子任务难度评估与模型分配 -> 目标方案, 最终方案
搜索得到最优解
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
from CHAMP_Trys.CHAMP_utils import *
from utils import *

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
openaiClient = setOpenAi(keyid = 4)
llamaClient = Groq(  # 这个是Groq调用llama的api接口
    api_key='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'
)
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "LLM分解任务到steps-20times-10_04"

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
    N = 100  # 对100个问题进行分解
    with open(file_path, 'r', encoding='utf-8') as file:
        problems = json.load(file)
    print(len(problems))
    sys.exit(0)
        
    doneids = extract_numbers_from_filenames('Decomposed_Steps_1004')
        
    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    MAX_TRY = 5  # 错误尝试最大值
    question_ids = list(range(0, N))
    decomTime = 20
    
    f = open('成功榜单.json', 'r')
    content = f.read()
    successIDs = json.loads(content)
    question_ids = [item for item in question_ids if item in successIDs and item not in doneids]
    
    # 选择问题
    for question_id in tqdm(question_ids):
        question = problems[question_id]['problem_text']
        gold_answer = problems[question_id]['problem_answer']
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('instruction content:')
        logger.info(question)
        
        results = []  # 每个问题一个分解步骤构成的list
        for de_time in tqdm(range(decomTime)):  # 每个问题分解20次,不然数据集量级根本不够.
            attempts = 0
            success = False
            while attempts < MAX_TRY and not success:  # 如果遇到格式错误
                try:
                    decompose_steps = decompose_sql(clients, question, config)
                    steps, steps_dict = convert_steps_to_format(decompose_steps)
                    relations_test = construct_dependencies_without_traversal(clients, question, steps, config)  # query LLM回答所有的依赖
                    G1 = create_dag_from_string(relations_test)
                    reduced_dependencies = list(G1.edges())
                    edges = []
                    for item in reduced_dependencies:
                        edges.append((item[0][:item[0].find('[')].strip(), item[1][:item[1].find('[')].strip()))
                    int_edges = [(int(e[0].split()[1]), int(e[1].split()[1])) for e in edges]
                    node_depths = calculate_node_depths(edges)
                    depths = reverseDict(node_depths)
                    
                    # Step 1 [ What does it mean for rooks to be placed "peacefully" on a chessboard? ] -> Step 3 [ How can we determine valid positions for a single rook such that the placement remains peaceful even after a 180-degree rotation of the board? ] 结果符合要求                          
                    results.append([steps, steps_dict, depths, int_edges])
                    # logger.info(f"分解成功!")
                    success = True
                    logger.info(f'\ndecompose time: {de_time}')
                except:
                    attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
                    logger.info(f"error: {attempts};  taskid: {question_id}")  # 生成过程出错了
            
            if attempts == MAX_TRY:
                error_Q += 1
                logger.info(f'run error {MAX_TRY}+')
            
            write_json_listoneline(f"Decomposed_Steps_1004/{question_id}.json", results)
            
        write_json(f"Decomposed_Steps_1004/{question_id}.json", results)
        logger.info(f"成功分解{len(results)}次")
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done! time cost: {elapsed_time}s")
    # print(f"#results: {len(results)}")
    
    # # 保存到本地文件
    # with open('Decom_Steps.pkl', 'wb') as f:
    #     pickle.dump(results, f)




