'''
需要把问题的prompts设计都与SCAN问题本身做适配
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
from SCAN_utils import *
from tqdm import tqdm

from utils import *

sys.path.append('C:\\Users\\Pluto\\Desktop\\TaDe')

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
openaiClient = setOpenAi(keyid = 0)
llamaClient = OpenAI(
    api_key="EMPTY",
    base_url="http://101.6.69.60:8000/v1",
)
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "gpt4o 最终方案测试 Step1"

if __name__ == '__main__':
    
    start_time = time.time()
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
    tokens_path = f'Tokens/token_usage_{formatted_now}.json'  # 这是记录token消耗的文件
    if not os.path.exists(tokens_path):
        with open(tokens_path, 'w') as f:
            json.dump({}, f)
    
    with open('SCAN_config.json', 'r') as f:
        config = json.load(f)
    logger, filename = setup_logger(aftername)
    config['tokens_path'] = tokens_path

    # 示例文件路径
    file_path = '../Task_Datasets/SCAN/SCAN_all_tasks.txt'
    N = 200
    count = 0
    tasks = []
    solutions = []
    # 打开文件并逐行读取内容
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:            
            question, actions = extract_in_out(line.strip())            
            tasks.append(question)
            actions = [action.replace("I_", "") for action in actions.split()]
            solutions.append(actions)
            if count == N:
                break

    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    question_ids = list(range(N))
    step1Res = {}
    
    # 选择问题
    for question_id in tqdm(question_ids):
        
        step1Res[question_id] = {}
        question = tasks[question_id]
        
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('instruction content:')
        logger.info(question)

        attempts = 0
        success = False
        while attempts < 5 and not success:  # 如果遇到格式错误
            try:
                # 问题分解
                decompose_steps = decompose_sql(clients, question, config)
                # decompose_steps: "walk opposite right thrice after run opposite right" can be solved by: "run opposite right", "walk opposite right thrice".
                # print(decompose_steps)  # 基本没有问题
                
                # 分解后格式规范化
                steps, steps_dict = convert_steps_to_format(decompose_steps)
                formatted_steps = '; '.join([f'step{i+1}: {step}' for i, step in enumerate(steps)])
                # commands_decomposed: ['run opposite right', 'walk opposite right thrice']
                # print(steps_dict)
                
                # 依赖性分析
                relations_test = construct_dependencies_without_traversal(clients, question, steps, config)  # query LLM回答所有的依赖
                # relations_test:  Step 2 [ run opposite right ] -> Step 1 [ walk opposite right thrice]
                # print('relations_test:\n', relations_test)
                
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
                depths = reverseDict(node_depths)  # {0: ['Step 1'], 1: ['Step 2'], 2: ['Step 3']}
                # print('深度计算 done')
                
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
        
        if attempts == 5:
            error_Q += 1
            logger.info('run error 5+')
            
    write_json_listoneline("step1Res_scan-last.json", step1Res)
    
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
    
    
    