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
openaiClient = setOpenAi(keyid = 3)
llamaClient = Groq(  # 这个是Groq调用llama的api接口
    api_key='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'
)
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "LLM自动化任务分配"

if __name__ == '__main__':
    successID = []
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
    N = 100
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
    
    # question_ids = [29, 48]

    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    question_ids = list(range(100))
    
    # 选择问题
    for question_id in tqdm(question_ids):
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
                # commands_decomposed: ['run opposite right', 'walk opposite right thrice']
                # print(steps_dict)
                
                # LLM自动化执行任务分配
                # allo_model = AllocateModel(clients, question, steps, config)  # 很好用
                allo_model =  {i: 'gpt-4-turbo' for i in range(1, len(steps)+1)}
                # print(allo_model)
                
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
                        sys_q = f"""There is a natural language instruction representing a sequence of actions. I need you to translate this sentence from natural language into a standardized meta-action sequence."
Here is the instruction:\n{question}

I have broken this instruction down into some smaller instructions. I will assign you sub-instructions one by one, and provide the results of the previous sub-instructions as a reference for your reasoning.
Please organize your reasoning according to the combination and progression of actions.

For your reference, 13 examples for translation together with the corresponding explanations are as follows:

Q: "turn left"
A: "turn left" outputs "TURN LEFT".

Q: "turn right"
A: "turn right" outputs "TURN RIGHT".

Q: "jump left"
A: The output of “jump left” concatenates: the output of “turn left”, the output of “jump”. “turn left” outputs “TURN LEFT”. “jump” outputs “JUMP”. So concatenating the output of “turn left” and the output of “jump” leads to “TURN LEFT” + “JUMP”. So the output of “jump left” is “TURN LEFT” + “JUMP”.

Q: "run right"
A: The output of "run right" concatenates: the output of "turn right", the output of "run". "turn right" outputs "TURN RIGHT". "run" outputs "RUN". So concatenating the output of "turn right" and the output of "run" leads to "TURN RIGHT" + "RUN". So the output of "run right" is "TURN RIGHT" + "RUN".

Q: "look twice"
A: The output of "look twice" concatenates: the output of "look", the output of "look". "look" outputs "LOOK". So repeating the output of "look" two times leads to "LOOK" * 2. So the output of "look twice" is "LOOK" * 2.

Q: "run and look twice"
A: The output of "run and look twice" concate+nates: the output of "run", the output of "look twice". "run" outputs "RUN". "look twice" outputs "LOOK" * 2. So concatenating the output of "run" and the output of "look twice" leads to "RUN" + "LOOK" * 2. So the output of "run and look twice" is "RUN" + "LOOK" * 2.

Q: "jump right thrice"
A: The output of "jump right thrice" concatenates: the output of "jump right", the output of "jump right", the output of "jump right". "jump right" outputs "TURN RIGHT" + "JUMP". So repeating the output of "jump right" three times leads to ("TURN RIGHT" + "JUMP") * 3. So the output of "jump right thrice" is ("TURN RIGHT" + "JUMP") * 3.

Q: "walk after run"
A: The output of "walk after run" concatenates: the output of "run", the output of "walk". "run" outputs "RUN". "walk" outputs "WALK". So concatenating the output of "run" and the output of "walk" leads to "RUN" + "WALK". So the output of "walk after run" is "RUN" + "WALK".

Q: "turn opposite left"
A: The output of "turn opposite left" concatenates: the output of "turn left", the output of "turn left". "turn left" outputs "TURN LEFT". So repeating the output of "turn left" twice leads to "TURN LEFT" * 2. So the output of "turn opposite left" is "TURN LEFT" * 2.

Q: "turn around left"
A: The output of "turn around left" concatenates: the output of "turn left", the output of "turn left", the output of "turn left", the output of "turn left". "turn left" outputs "TURN LEFT". So repeating the output of "turn left" four times leads to "TURN LEFT" * 4. So the output of "turn around left" is "TURN LEFT" * 4. Q: "turn opposite right" A: The output of "turn opposite right" concatenates: the output of "turn right", the output of "turn right". "turn right" outputs "TURN RIGHT". So repeating the output of "turn right" twice leads to "TURN RIGHT" * 2. So the output of "turn opposite right" is "TURN RIGHT" * 2.

Q: "turn around right"
A: The output of "turn around right" concatenates: the output of "turn right", the output of "turn right", the output of "turn right", the output of "turn right". "turn right" outputs "TURN RIGHT". So repeating the output of "turn right" four times leads to "TURN RIGHT" * 4. So the output of "turn around right" is "TURN RIGHT" * 4.

Q: "walk opposite left"
A: The output of "walk opposite left" concatenates: the output of "turn opposite left", the output of "walk". "turn opposite left" outputs "TURN LEFT" * 2. "walk" outputs "WALK". So concatenating the output of "turn opposite left" and the output of "walk" leads to "TURN LEFT" * 2 + "WALK". So the output of "walk opposite left" is "TURN LEFT" * 2 + "WALK".

Q: "walk around left"
A: The output of "walk around left" concatenates: the output of "walk left", the output of "walk left", the output of "walk left", the output of "walk left". "walk left" outputs "TURN LEFT" + "WALK". So repeating the output of "walk around left" four times leads to ("TURN LEFT" + "WALK") * 4. So the output of "walk around left" is ("TURN LEFT" + "WALK") * 4.

Please pay attention to the use of parentheses.
"""  # 系统任务信息
                        
                        if len(answerDict)>0:
                            answersSoFar = f"""\nSo far, the answers to the resolved sub-instructions are as follows: The format is Sub-instruction-Id: xxx; Sub-instruction: xxx; Answer: xxx."""
                            for key, value in answerDict.items():
                                answersSoFar += f"""\nSub-instruction-Id: {key}; Sub-instruction: {answerDict[key]['subtask']}; Answer: {answerDict[key]['answer']}."""
                            
                            predecessors = search_Predecessors(int_edges, number)
                            intersection = set(answerDict.keys()).intersection(set(predecessors))
                            count = len(intersection)
                            if count>0:
                                answersSoFar += f"""\nAmong them, sub-instructions {predecessors} are directly related to this sub-instruction, so please pay special attention to them."""
                        
                        
                        subask = f"""\nThe sub-instruction to be converted now is:: {subtask}
Based on the information above, please provide a concise and clear answer"""

                        if len(answerDict)>0:
                            query = answersSoFar+subask
                        else:
                            query = subask

                        Q = [{'role':'system', 'content':sys_q},
                            {'role':'user', 'content':query},]
                            
                        result = askLLM(clients, Q, tokens_path=tokens_path, model=answer_MODEL, temperature=1, max_tokens=300)

                        answerDict[number] = {'subtask':subtask, 'answer':result}
                        progress_bar.update(1)

                progress_bar.close()

                # 已经问完了所有的subtask,最后问一次得到最终的答案
                Q.append({'role':'assistant', 'content':result})
                Q.append({'role':'user', 'content':f"""Now that all the sub-instructions have been completed, so what is the final answer?
Please give the final action sequence without any additional explanation or clarification."""})
                # finalResult = askChatGPT(Q, model=config["finalSummarize_MODEL"], temperature=1)
                finalResult = askLLM(clients, Q, tokens_path=tokens_path, model=config['finalSummarize_MODEL'], temperature=1)
                # print('图上推理 done')
                
                # 现在已经问题不大了.
                actionSeq = sentenceRes2Actions(clients, finalResult, config)
                actionList = actionSeq.split()
                
                logger.info('answer: '+str(actionList))
                logger.info('gold: '+str(solutions[question_id]))
                

                # 理论上结果直接和真实结果对比一下,算个正确率即可
                if actionList == solutions[question_id]:
                    success_Q += 1
                    logger.info('correct')
                    successID.append(question_id)
                    with open('成功榜单.json', 'w') as file:
                        json.dump(successID, file, indent=4)
                else:
                    unsuccess_Q += 1
                    logger.info('error')
                    
                success = True  # 任务未受中断,完整地结束了,所以标记为成功

            except:
                attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
                print(f"error: {attempts};  taskid: {question_id}")  # 生成过程出错了
        
        if attempts == 5:
            error_Q += 1
            logger.info('run error 5+')
    
    end_time = time.time()
    # 计算运行时间
    elapsed_time = end_time - start_time
    hours, minutes, seconds = seconds_to_hms(elapsed_time)
    logger.info(f"100 solving 运行耗时: {hours}h, {minutes}min, {seconds}s")
       
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
    
    
    