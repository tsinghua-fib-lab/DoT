'''
CSQA 常识推理问题,问题大都比较简单
问题简单,注意不要让LLM把问题分解得太复杂了
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
from CSQA_Trys.CSQA_utils import *
from utils import *

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
openaiClient = setOpenAi(keyid = 0)
llamaClient = Groq(  # 这个是Groq调用llama的api接口
    api_key='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'
)
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "run_gpt-4-turbo统一解决"

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
    
    with open('CSQA_config.json', 'r') as f:
        config = json.load(f)
    config['tokens_path'] = tokens_path
        
    file_path = '../Task_Datasets/CSQA/train_rand_split.jsonl'
    questions = []
    with open(file_path, 'r') as f:
        # 逐行读取文件内容
        for line in f:
            # 将每一行转换为JSON对象
            entry = json.loads(line)
            # 处理或存储JSON对象
            questions.append(entry)

    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    N = 100
    question_ids = list(range(85, N))
    ERROR_ATTEMPTS = 5
    
    # 选择问题
    for question_id in tqdm(question_ids):
        
        question = questions[question_id]['question']['stem']  # 应该把选项也放进问题里,作为问题的一部分.
        options = questions[question_id]['question']['choices']  # CSQA数据集是给出具体选项的
        gold_answer = questions[question_id]['answerKey']
        
        # 转换为所需的字符串格式
        options_string = "; ".join([f"{item['label']}: {item['text']}" for item in options])
            
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('problem content:\n')
        logger.info(question)

        attempts = 0
        success = False
        while attempts < ERROR_ATTEMPTS and not success:  # 如果遇到格式错误
            try:
                # 问题分解
                decompose_steps = decompose_sql(clients, question, options_string, config)
                # decompose_steps:
                # print(decompose_steps)
                # sys.exit(0)
                # 1. Define the meaning of "sanctions" in the context of this sentence
                # print(decompose_steps)
                # print('\n')
                 
                # 分解后格式规范化
                steps, steps_dict = convert_steps_to_format(decompose_steps)
                # print(steps)
                # print('\n')
                # print(steps_dict)
                # print('\n')
                # 注意steps_dict里的key是纯数字.
                
                # LLM自动化执行任务分配
                # allo_model = AllocateModel(question, steps, config)  # 很好用
                # print(allo_model)
                
                allo_model = {key: 'gpt-4-turbo' for key in steps_dict.keys()}
                # 先统一用gpt-4-turbo来试一下结果如何
                
                # 依赖性分析
                relations_test = construct_dependencies_without_traversal(clients, question, options_string, steps, config)  # query LLM回答所有的依赖
                # relations_test:  Step 2 [ run opposite right ] -> Step 1 [ walk opposite right thrice]
                
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

                # 开始基于图进行推理
                heights = list(depths.keys())
                MAXHeight = max(heights)
                answerDict = {}  # 只有已经做过回答的subtask才会被放到这里面来
                progress_bar = tqdm(total=len(steps))
                for i in range(MAXHeight):
                    subtasks = depths[i]
                    for subtaskid in subtasks:                
                        number = re.findall(r'\d+', subtaskid)
                        number = int(number[0]) if number else None
                        subtask = steps_dict[number]
                        answer_MODEL = allo_model[number]
                        
                        # question 问题字符串
                        # 交待解决任务
                        sys_q = f"""There is a single-choice question involving common sense reasoning. I need you to solve it and give the right answer.
Here is the question:\n{question} 
Here are the options: \n{options_string}

I have broken this common sense reasoning question down into several smaller questions. I will assign you sub-questions one by one, and provide the results of the previous sub-questions as a reference for your reasoning."""  # 系统任务信息
                        
                        if len(answerDict)>0:
                            answersSoFar = f"""\nSo far, the answers to the resolved sub-questions are as follows: The format is Sub-question-Id: xxx; Sub-question: xxx; Answer: xxx."""
                            for key, value in answerDict.items():
                                answersSoFar += f"""\nSub-question-Id: {key}; Sub-question: {answerDict[key]['subtask']}; Answer: {answerDict[key]['answer']}."""
                            
                            predecessors = search_Predecessors(int_edges, number)
                            intersection = set(answerDict.keys()).intersection(set(predecessors))
                            count = len(intersection)
                            if count>0:
                                answersSoFar += f"""\nAmong them, sub-questions {predecessors} are directly related to this sub-question, so please pay special attention to them."""
                        
                        
                        subask = f"""\nThe sub-question to solve now is xxx: {subtask}
Based on the information above, please provide a concise and clear answer"""

                        if len(answerDict)>0:
                            query = answersSoFar+subask
                        else:
                            query = subask

                        Q = [{'role':'system', 'content':sys_q},
                            {'role':'user', 'content':query},]
                        
                        result = askLLM(clients, Q, tokens_path=tokens_path, model=answer_MODEL, temperature=1, max_tokens=300)
                        # result = askChatGPT(Q, model='gpt-3.5-turbo', temperature=1, max_tokens=300)
                        # print('\n\n\n')
                        # print('Question:',subtask)
                        # print('\n\n\n')
                        # print('Answer:', result)
                        # print('\n\n\n')
                        # sys.exit(0)
                        
                        answerDict[number] = {'subtask':subtask, 'answer':result}
                        progress_bar.update(1)

                progress_bar.close()

                # 已经问完了所有的subtask,最后问一次得到最终的答案
                Q.append({'role':'assistant', 'content':result})
                Q.append({'role':'user', 'content':f"""Now that all the sub-questions have been solved, which answer do you ultimately choose?
Please provide only the letter of the option, without any additional explanation or description."""})
                # finalResult = askChatGPT(Q, model=GPT_MODEL, temperature=1)
                finalResult = askLLM(clients, Q, tokens_path=tokens_path, model=config['finalSummarize_MODEL'], temperature=1, max_tokens=300)
                # print('最终回答 done')
                
                logger.info('finalResult: ')
                logger.info(finalResult)
                finalResult = finalResult.strip()
                
                
                if finalResult == gold_answer:
                    success_Q += 1
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功
                    logger.info('correct')   
                else:
                    unsuccess_Q += 1
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功   
                    logger.info('error') 
            
            except:
                attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
                logger.info(f"error: {attempts};  taskid: {question_id}")  # 生成过程出错了
        
        if attempts == ERROR_ATTEMPTS:
            error_Q += 1
            logger.info(f'run error {ERROR_ATTEMPTS}+')


    end_time = time.time()
    # 计算运行时间
    elapsed_time = end_time - start_time
    hours, minutes, seconds = seconds_to_hms(elapsed_time)
    logger.info(f"100 solving 运行耗时: {hours}h, {minutes}min, {seconds}s")
    
    logger.info(f'\n{tokens_path}')
    logger.info(f'Correct_Q: {success_Q}')
    logger.info(f'False_Q: {unsuccess_Q}')
    logger.info(f'Error_Q: {error_Q}')
    
    # 读取文件并打印结果以验证
    with open(tokens_path, 'r') as f:
        token_usage = json.load(f)
        # logger.info(json.dumps(token_usage, indent=4))
        total_tokens, total_cost = CountCost(token_usage)
        # 打印结果
        logger.info(f"Total Tokens: {total_tokens}")
        logger.info(f"Total Cost: ${total_cost:.2f}")
            
    
    