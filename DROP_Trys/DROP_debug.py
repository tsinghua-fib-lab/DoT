'''
MATH数学问题
在least2most里没有设计好的prompts了
自己寻找适配CHAMP数据集的prompts设计
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
from typing import List

import numpy as np
import openai
from tqdm import tqdm

sys.path.append('C:\\Users\\Pluto\\Desktop\\TaDe')
from DROP_Trys.DROP_utils import *
from utils import *

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
setOpenAi(keyid = 3)

if __name__ == '__main__':
    
    logger, filename = setup_logger(GPT_MODEL)

    # 示例文件路径
    file_path = '../Task_Datasets/DROP/drop_100.json'
    N = 100
    
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
    question_ids = list(range(100))
    
    # 选择问题
    for question_id in tqdm(question_ids):
        
                passage = problems[question_id]['passage']
                question = problems[question_id]['question']
                gold_answer = problems[question_id]['answer']
                
                logger.info('\n\n\n')
                logger.info(f'number id: {question_id}')
                logger.info('instruction content:')
                logger.info(question)

        # attempts = 0
        # success = False
        # while attempts < 3 and not success:  # 如果遇到格式错误
        #     try:
                # 问题分解
                decompose_steps = decompose_sql(passage, question)
                # decompose_steps:
                # print('\n\n\n')
                # print(decompose_steps)
                # sys.exit(0)
                 
                # 分解后格式规范化
                steps, steps_dict = convert_steps_to_format(decompose_steps)
                # print(steps)
                # print(steps_dict)
                # sys.exit(0)
                
                # 依赖性分析
                relations_test = construct_dependencies_without_traversal(question, steps)  # query LLM回答所有的依赖
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
                        
                        # question 问题字符串
                        # 交待解决任务
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
                            
                        result = askChatGPT(Q, model=GPT_MODEL, temperature=1, max_tokens=300)  # 主动限制回答长度
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
                Q.append({'role':'user', 'content':f"""Now that all the sub-questions have been solved, so what is the final answer?
Please give the final answer without any additional explanation or clarification."""})
                finalResult = askChatGPT(Q, model=GPT_MODEL, temperature=1)
                # print('图上推理 done')
                
                logger.info('LLM Answer: '+finalResult)
                # print('\n\nFinalResult:')
                # print(finalResult)
                # sys.exit(0)
                
                # 让大语言模型来判断有没有回答正确
                judgeAnswer = {'role':'user', 'content':f"""Here is a math problem with a standard answer and a student's solution. Please help me determine if the student's solution is correct.
Problem: {question}

Standard answer: {gold_answer}

Answer: {finalResult}

If the student's answer is correct, just output True; otherwise, just output False.
No explanation is required.
"""}
                Q_judge = [judgeAnswer]
                ifcorrect = askChatGPT(Q_judge, model=GPT_MODEL, temperature=1)  # 要么是True, 要么是False
                
                if 'True' in ifcorrect:
                    success_Q += 1
                    print('correct')
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功
                elif 'False' in ifcorrect:
                    unsuccess_Q += 1
                    print('error')
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功                    
                
            # except:
            #     attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
            #     print(f"error: {attempts};  taskid: {question_id}")  # 生成过程出错了
        
        # if attempts == 3:
        #     error_Q += 1
        #     logger.info('run error 3+')
          
    logger.info(f'correct_Q: {success_Q}')
    logger.info(f'error_Q: {error_Q}')
    
    
    