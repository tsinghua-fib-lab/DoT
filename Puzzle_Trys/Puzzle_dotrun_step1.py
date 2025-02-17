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
import logging

from puzzle_utils import *

from utils import *

# client定义需要满足如下调用方式: client.chat.completions.create(model,messages = messages), 详见askLLM函数
openaiClient = setOpenAi(keyid = 0)
llamaClient = setLocal()
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "final_version-step1"

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
    # 选择问题
    
    step1Res = {}
    N = 200
    MAX_TRY = 5
    
    for question_id in tqdm(range(N)):
                # question_id = random.randint(0, len(puzzles))
                # question_id = 519
        step1Res[question_id] = {}
        question = puzzles[question_id]['sat']
        
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('label id: '+puzzles[question_id]['name'])
        logger.info('puzzle content:')
        logger.info(question)

        attempts = 0
        success = False
        while attempts < MAX_TRY and not success:  # 如果遇到格式错误
            try:
                # 问题分解
                decompose_steps = decompose_sql(clients, question, config)
                # 分解后格式规范化
                steps, steps_dict = convert_steps_to_format(decompose_steps)
                # print('问题分解 done')

                # 依赖性分析
                relations_test = construct_dependencies_without_traversal(clients, question, steps, config)  # query LLM回答所有的依赖
                # 建图与化简
                G1 = create_dag_from_string(relations_test)
                reduced_dependencies = list(G1.edges())
                # 边形式化简
                edges = []
                for item in reduced_dependencies:
                    edges.append((item[0][:item[0].find('[')].strip(), item[1][:item[1].find('[')].strip()))
                int_edges = [(int(e[0].split()[1]), int(e[1].split()[1])) for e in edges]
                # print('建图 done')

                # 计算节点的深度
                node_depths = calculate_node_depths(edges)
                # 按照深度重新组织节点
                depths = reverseDict(node_depths)  # {0: ['Step 1'], 1: ['Step 3', 'Step 2'], 2: ['Step 5', 'Step 4'], 3: ['Step 6'], 4: ['Step 7'], 5: ['Step 8'], 6: ['Step 9']}
                # print('深度计算 done')
                
                formatted_steps = '; '.join(['step{}: {}'.format(i, step['step']) for i, step in enumerate(steps)])
                steps = [x['step'] for x in steps]
                
                step1Res[question_id]['steps'] = steps
                step1Res[question_id]['steps_dict'] = steps_dict
                step1Res[question_id]['depths'] = depths
                step1Res[question_id]['int_edges'] = int_edges
                
                step1Res[question_id]['problemText'] = question
                step1Res[question_id]['allSubtask'] = formatted_steps
                step1Res[question_id]['nowSubtask'] = steps  # 这是一个list吧

                success = True  # 任务未受中断,完整地结束了,所以标记为成功
            
            except:
                attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
                print(f"error: {attempts};  taskid: {question_id}")  # 生成过程出错了
                
        if attempts == MAX_TRY:
            error_Q += 1
            logger.info('run error 5+')
    
    write_json_listoneline("step1Res_puzzle-last.json", step1Res)
            
    end_time = time.time()
    # 计算运行时间
    elapsed_time = end_time - start_time
    hours, minutes, seconds = seconds_to_hms(elapsed_time)
    logger.info(f"100 solving 运行耗时: {hours}h, {minutes}min, {seconds}s")     
    
    # 读取文件并打印结果以验证
    with open(tokens_path, 'r') as f:
        token_usage = json.load(f)
        # logger.info(json.dumps(token_usage, indent=4))
        total_tokens, total_cost = CountCost(token_usage)
        # 打印结果
        logger.info(f"Total Tokens: {total_tokens}")
        logger.info(f"Total Cost: ${total_cost:.2f}")
    
    
    