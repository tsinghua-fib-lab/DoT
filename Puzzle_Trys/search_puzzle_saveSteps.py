'''
MATH数学问题
在least2most里没有设计好的prompts了
自己寻找适配CHAMP数据集的prompts设计
'''
'''
Here is a math word problem. I will first provide a description of the problem to set the context. Then, I will ask a specific question that requires you to use the information from the problem description, along with calculation and reasoning, to solve it.
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
from Puzzle_Trys.puzzle_utils import *
from utils import *

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
openaiClient = setOpenAi(keyid = 1)  # 注意不要一个key多个代码使用
llamaClient = Groq(  # 这个是Groq调用llama的api接口
    api_key='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'
)
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "LLM分解任务到steps-20times-10_07"

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
    

    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    N = 100
    MAX_TRY = 5  # 错误尝试最大值
    DECOM_TIME = 20
    question_ids = list(range(0, N))
    
    doneids = extract_numbers_from_filenames('Decomposed_Steps_1004')
    f = open('成功榜单.json', 'r')
    content = f.read()
    successIDs = json.loads(content)
    question_ids = [item for item in question_ids if item in successIDs and item not in doneids]
    
    results = []
        
    # 选择问题
    for question_id in tqdm(question_ids):
        
        question = puzzles[question_id]['sat']
        
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('problem content:\n')
        logger.info(question)
        
        results = []
        for de_time in range(DECOM_TIME):  # 每个问题分解8次,由于时间的制约,先分解8次再说.
            attempts = 0
            success = False
            while attempts < MAX_TRY and not success:  # 如果遇到格式错误
                try:
                    decompose_steps = decompose_sql(clients, question, config)
                    steps, steps_dict = convert_steps_to_format(decompose_steps)
                    relations_test = construct_dependencies_without_traversal(clients, question, steps, config)  # query LLM回答所有的依赖
                    G1 = create_dag_from_string(relations_test)
                    reduced_dependencies = list(G1.edges())
                    # 边形式化简
                    edges = []
                    for item in reduced_dependencies:
                        edges.append((item[0][:item[0].find('[')].strip(), item[1][:item[1].find('[')].strip()))
                    int_edges = [(int(e[0].split()[1]), int(e[1].split()[1])) for e in edges]

                    # 计算节点的深度
                    node_depths = calculate_node_depths(edges)
                    depths = reverseDict(node_depths)
                    depthlen = sum(len(value) for value in depths.values())
                    if len(steps) == depthlen:
                        results.append([steps, steps_dict, depths, int_edges])
                        success = True  # 任务未受中断,完整地结束了,所以标记为成功
                        logger.info(f'\ndecompose time: {de_time}')
                except:
                    attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
                    logger.info(f"error: {attempts};  taskid: {question_id}")  # 生成过程出错了
            
            if attempts == MAX_TRY:
                logger.info(f'run error {MAX_TRY}+')
        
        write_json(f"Decomposed_Steps_1004/{question_id}.json", results)
        logger.info(f"成功分解{len(results)}次")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done! time cost: {elapsed_time}s")
            
    
    