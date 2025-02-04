'''
run2: 根据回答的alpha-quantile来对模型进行升级
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
from DROP_Trys.DROP_utils import *
from utils import *

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
openaiClient = setOpenAi(keyid = 0)
llamaClient = Groq(  # 这个是Groq调用llama的api接口
    api_key='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'
)
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "gpt4o 任务两阶段 第一阶段"

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
    with open('DROP_config.json', 'r') as f:
        config = json.load(f)
    config['tokens_path'] = tokens_path

    # 示例文件路径
    file_path = 'C:\\Users\Pluto\Desktop\TaDe\Task_Datasets\DROP\\all_drop_p.json'
    
    with open(file_path, 'r', encoding='utf-8') as file:
        problems = json.load(file)
        
    # 先给出文本,再给出问题,最后给出答案    
    # print(len(problems))
    # print('\n\n')
    # print(problems[0]['passage'])
    # print('\n\n')
    # print(problems[0]['question'])
    # print('\n\n')
    # print(problems[0]['answer'])
        
    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0

    N = 200
    question_ids = list(range(N))
    
    step1Res = {}
    
    MAX_TRY = 5  # 错误尝试最大值
    SUB_UPGRADE_UPPER = 2  # 每个子问题最多的LLM升级次数 
    update_subtask = 0  # 记录了一共有多少个子任务用到了升级
    update_times = 0  # 记录了LLM模型选择一共升级了多少次
    num_subtasks = 0  # 记录了一共分解出了多少个子任务
    
    # 选择问题
    for question_id in tqdm(question_ids):
        
        step1Res[question_id] = {}
        
        passage = problems[question_id]['passage']
        question = problems[question_id]['question']
        gold_answer = problems[question_id]['answer']
        
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('instruction content:')
        logger.info(question)

        attempts = 0
        success = False
        while attempts < MAX_TRY and not success:  # 如果遇到格式错误
            try:
                # 问题分解
                this_update_subtask = 0  # 记录了一共有多少个子任务需要处理
                this_update_times = 0  # 记录了LLM模型选择一共升级了多少次
                
                # 问题分解
                decompose_steps = decompose_sql(clients, passage, question, config)
                # decompose_steps:
                # print('\n\n\n')
                # print(decompose_steps)
                 
                # 分解后格式规范化
                steps, steps_dict = convert_steps_to_format(decompose_steps)
                formatted_steps = '; '.join([f'step{i+1}: {step}' for i, step in enumerate(steps)])
                # print(steps)
                # print(steps_dict)
                
                # 依赖性分析
                relations_test = construct_dependencies_without_traversal(clients, question, steps, config)  # query LLM回答所有的依赖
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
                
                step1Res[question_id]['problemText'] = passage + '\n' + question
                step1Res[question_id]['allSubtask'] = formatted_steps
                step1Res[question_id]['nowSubtask'] = steps  # 这是一个list吧
                
                success = True  # 任务未受中断,完整地结束了,所以标记为成功
                
            except:
                attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
                logger.info(f"error: {attempts};  taskid: {question_id}")  # 生成过程出错了
        
        if attempts == MAX_TRY:
            error_Q += 1
            logger.info(f'run error {MAX_TRY}+')
    
    write_json_listoneline("step1Res_DROP-last.json", step1Res)
            
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
    
    
    