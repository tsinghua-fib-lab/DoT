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
llamaClient = OpenAI(
    api_key="EMPTY",
    base_url="http://101.6.69.60:8002/v1",
)
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "最终方案结果-第二阶段结果"

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
        
    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0

    N = 200
    question_ids = list(range(N))
    
    MAX_TRY = 8  # 错误尝试最大值
    
    f = open('TmpRes/step2In_DROP_last.json', 'r')
    content = f.read()
    middleRes = json.loads(content) 
    
    # 选择问题
    for question_id in tqdm(question_ids):
        
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
                steps, steps_dict, allo_model, depths, int_edges = middleRes[str(question_id)]['steps'], middleRes[str(question_id)]['steps_dict'], middleRes[str(question_id)]['allo_model'], middleRes[str(question_id)]['depths'], middleRes[str(question_id)]['int_edges']
                depths = {int(k): v for k, v in depths.items()}

                # 开始基于图进行推理
                heights = list(depths.keys())
                MAXHeight = max(heights)
                answerDict = {}  # 只有已经做过回答的subtask才会被放到这里面来
                progress_bar = tqdm(total=len(steps))
                for number, step in enumerate(steps):               
                    subtask = steps_dict[str(number+1)]
                    answer_MODEL = allo_model[number]
                        
                    sys_q = f"""Here is a math word problem. I will first provide a passage of the problem to set the context. Then, I will ask a specific question that requires you to use the information from the problem description, along with calculation and reasoning, to solve it. 
Passage:\n{passage}

Question:\n{question}

I have broken this math question down into several smaller questions. I will assign you sub-questions one by one, and provide the results of the previous sub-questions as a reference for your reasoning.
Please solve the question according to mathematical logic.
    """  # 系统任务信息
                    
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
                    answerDict[number] = {'subtask':subtask, 'answer':result}
                    progress_bar.update(1)

                progress_bar.close()

                # 已经问完了所有的subtask,最后问一次得到最终的答案
                Q.append({'role':'assistant', 'content':result})
                Q.append({'role':'user', 'content':f"""Now that all the sub-questions have been solved, so what is the final answer?
Please give the final answer without any additional explanation or clarification."""})
                # finalResult = askChatGPT(Q, model=GPT_MODEL, temperature=1)
                finalResult = askLLM(clients, Q, tokens_path=tokens_path, model=config['finalSummarize_MODEL'], temperature=1, max_tokens=300)
                # print('图上推理 done')
                
                logger.info('LLM Answer: '+finalResult)
                # 让大语言模型来判断有没有回答正确
                judgeAnswer = {'role':'user', 'content':f"""Here is a math problem with a standard answer and a student's solution. Please help me determine if the student's solution is correct.
Problem: {question}

Standard answer: {gold_answer}

Answer: {finalResult}

If the student's answer is correct, just output True; otherwise, just output False.
No explanation is required.
"""}
                Q_judge = [judgeAnswer]
                # ifcorrect = askChatGPT(Q_judge, model=GPT_MODEL, temperature=1)  # 要么是True, 要么是False
                ifcorrect = askLLM(clients, Q_judge, tokens_path=tokens_path, model=config['judgeCorrect_MODEL'], temperature=1)
                
                if 'True' in ifcorrect:
                    success_Q += 1
                    logger.info('correct')
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功
                elif 'False' in ifcorrect:
                    unsuccess_Q += 1
                    logger.info('error')
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功            
                
            except:
                attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
                logger.info(f"error: {attempts};  taskid: {question_id}")  # 生成过程出错了
        
        if attempts == MAX_TRY:
            error_Q += 1
            logger.info(f'run error {MAX_TRY}+')
            
            
    end_time = time.time()
    # 计算运行时间
    elapsed_time = end_time - start_time
    hours, minutes, seconds = seconds_to_hms(elapsed_time)
    logger.info(f"100 solving 运行耗时: {hours}h, {minutes}min, {seconds}s")
    
    logger.info(f'\n{tokens_path}')
    logger.info(f'Correct_Q: {success_Q}')
    logger.info(f'False_Q: {unsuccess_Q}')
    logger.info(f'Error_Q: {error_Q}\n')
    
    # 读取文件并打印结果以验证
    with open(tokens_path, 'r') as f:
        token_usage = json.load(f)
        # logger.info(json.dumps(token_usage, indent=4))
        total_tokens, total_cost = CountCost(token_usage)
        # 打印结果
        logger.info(f"Total Tokens: {total_tokens}")
        logger.info(f"Total Cost: ${total_cost:.2f}")
    
    
    