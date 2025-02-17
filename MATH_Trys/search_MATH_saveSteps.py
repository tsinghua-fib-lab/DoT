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
from MATH_Trys.MATH_utils import *
from utils import *

# client定义需要满足如下调用方式: client.chat.completions.create(model,messages = messages), 详见askLLM函数
openaiClient = setOpenAi(keyid = 0)
llamaClient = setLocal()
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "LLM分解任务到steps-20times-10_05"

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
    
    with open('MATH_config.json', 'r') as f:
        config = json.load(f)
    config['tokens_path'] = tokens_path
        
    # file_path = '../Task_Datasets/MATH/math200.json'
    file_path = '../Task_Datasets/MATH/all_math_p.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        problems = json.load(file)
    print(len(problems))
    sys.exit(0)

    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    N = 400
    MAX_TRY = 5  # 错误尝试最大值
    DECOM_TIME = 20
    question_ids = list(range(0, N))
    
    doneids = extract_numbers_from_filenames('Decomposed_Steps_1004')
    f = open('成功榜单.json', 'r')
    content = f.read()
    successIDs = json.loads(content)
    question_ids = [item for item in question_ids if item in successIDs and item not in doneids]
    
    results = []
    
    question_ids = [0]
        
    # 选择问题
    for question_id in tqdm(question_ids):
        
        question = problems[question_id]['problem']
        type = problems[question_id]['type']
        gold_answer = problems[question_id]['solution']
        
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('problem content:\n')
        logger.info(question)
        
        results = []
        for de_time in range(DECOM_TIME):  # 每个问题分解8次,由于时间的制约,先分解8次再说.
            attempts = 0
            success = False
            while attempts < MAX_TRY and not success:  # 如果遇到格式错误
                # try:
                    decompose_steps = decompose_sql(clients, question, type, config)
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
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功
                    logger.info(f'\ndecompose time: {de_time}')

        write_json(f"Decomposed_Steps_1004/{question_id}.json", results)
        logger.info(f"成功分解{len(results)}次")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done! time cost: {elapsed_time}s")