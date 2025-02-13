'''
MATH数学问题
在least2most里没有设计好的prompts了
自己寻找适配CHAMP数据集的prompts设计
使用大语言模型执行子任务难度评估与模型分配 -> 目标方案, 最终方案
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
openaiClient = setOpenAi(keyid = 0)
llamaClient = Groq(  # 这个是Groq调用llama的api接口
    api_key='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'
)
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "gpt4o 最终方案测试 Step1"

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
    file_path = 'C:\\Users\Pluto\Desktop\TaDe\Task_Datasets\CHAMP\\all_champ_p.json'
    N = 200 
    with open(file_path, 'r', encoding='utf-8') as file:
        problems = json.load(file)

    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    MAX_TRY = 5  # 错误尝试最大值
    question_ids = list(range(N))
    
    step1Res = {}
    
    # 选择问题
    for question_id in tqdm(question_ids):
        
        step1Res[question_id] = {}
        question = problems[question_id]['problem_text']
        gold_answer = problems[question_id]['problem_answer']
        
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('instruction content:')
        logger.info(question)

        attempts = 0
        success = False
        while attempts < MAX_TRY and not success:  # 如果遇到格式错误
            try:
                decompose_steps = decompose_sql(clients, question, config)
                steps, steps_dict = convert_steps_to_format(decompose_steps)
                formatted_steps = '; '.join([f'step{i+1}: {step}' for i, step in enumerate(steps)])
                
                # 依赖性分析、建图
                relations_test = construct_dependencies_without_traversal(clients, question, steps, config)  # query LLM回答所有的依赖
                G1 = create_dag_from_string(relations_test)
                reduced_dependencies = list(G1.edges())
                edges = []
                for item in reduced_dependencies:
                    edges.append((item[0][:item[0].find('[')].strip(), item[1][:item[1].find('[')].strip()))
                int_edges = [(int(e[0].split()[1]), int(e[1].split()[1])) for e in edges]
                node_depths = calculate_node_depths(edges)
                # 按照深度重新组织节点
                depths = reverseDict(node_depths)
                
                # 任务分配
                # allo_model = AllocateModel(clients, question, steps, config)  
                
                step1Res[question_id]['steps'] = steps
                step1Res[question_id]['steps_dict'] = steps_dict
                step1Res[question_id]['depths'] = depths
                step1Res[question_id]['int_edges'] = int_edges
                
                step1Res[question_id]['problemText'] = question
                step1Res[question_id]['allSubtask'] = formatted_steps
                step1Res[question_id]['nowSubtask'] = steps  # 这是一个list吧
                
                success = True  # 任务未受中断,完整地结束了,所以标记为成功
                
                # self.problemText = [item["problemText"] for item in dataset]
                # self.allSubtask = [item["allSubtask"] for item in dataset]
                # self.nowSubtask = [item["nowSubtask"] for item in dataset]                     
                
            except:
                attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
                logger.info(f"error: {attempts};  taskid: {question_id}")  # 生成过程出错了
        
        if attempts == MAX_TRY:
            error_Q += 1
            logger.info(f'run error {MAX_TRY}+')
    
    write_json_listoneline("step1Res-last.json", step1Res)
            
    end_time = time.time()
    # 计算运行时间
    elapsed_time = end_time - start_time
    # print(f"程序运行耗时: {elapsed_time} 秒")
    hours, minutes, seconds = seconds_to_hms(elapsed_time)
    logger.info(f"200 solving 运行耗时: {hours}h, {minutes}min, {seconds}s")
    logger.info(f'\n{tokens_path}')
    logger.info(f'correct_Q: {success_Q}')
    logger.info(f'error_Q: {error_Q}')

    # 读取文件并打印结果以验证
    with open(tokens_path, 'r') as f:
        token_usage = json.load(f)
        # logger.info(json.dumps(token_usage, indent=4))
        total_tokens, total_cost = CountCost(token_usage)
        # 打印结果
        logger.info(f"Total Tokens: {total_tokens}")
        logger.info(f"Total Cost: ${total_cost:.2f}")
    
    
    
