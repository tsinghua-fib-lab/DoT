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
from typing import List

import numpy as np
import openai
from tqdm import tqdm

sys.path.append('C:\\Users\\Pluto\\Desktop\\TaDe')
from CHAMP_Trys.CHAMP_utils import *
from utils import *

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
setOpenAi(keyid = 3)

import json

with open('CHAMP_config.json', 'r') as f:
    config = json.load(f)
    
if __name__ == '__main__':

    # 示例文件路径
    file_path = '../Task_Datasets/CHAMP/champ_116.json'
    N = 100
    
    with open(file_path, 'r', encoding='utf-8') as file:
        problems = json.load(file)
    # print(len(problems))
    # print('\n\n')
    # print(problems[0]['problem_text'])
    # print('\n\n')
    # print(problems[0]['problem_answer'])

    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    question_ids = list(range(100))
    
    # 选择问题
    for question_id in tqdm(question_ids):
        
                question = problems[question_id]['problem_text']
                gold_answer = problems[question_id]['problem_answer']
                
                print('\n\n\n')
                print(f'number id: {question_id}')
                print('instruction content:')
                print(question)

        # attempts = 0
        # success = False
        # while attempts < 3 and not success:  # 如果遇到格式错误
        #     try:
                # 问题分解
                decompose_steps = decompose_sql(question, config)
                # decompose_steps:
                """
                1. What does it mean for rooks to be placed peacefully on a chessboard?
                2. What does 180-degree rotational invariance imply for the placement of the rooks?  
                3. How does the property of 180-degree rotational invariance constrain possible placements on the board?
                4. How is the total number of ways to place n rooks peacefully on an n x n board calculated without the rotational constraint?
                5. How can we calculate or enumerate the placements that also satisfy the 180-degree 
                rotational constraint?
                6. Do the potential placements differ based on whether n is even or odd?
                7. How does one verify that a specific arrangement of rooks is unique and obeys all given constraints?
                """
                # print('\n\n\n')
                # print(decompose_steps)
                 
                # 分解后格式规范化
                steps, steps_dict = convert_steps_to_format(decompose_steps)                
                # print(steps)
                # print(steps_dict)
                # Step 1 [ What does it mean for rooks to be placed "peacefully" on a chessboard? ] -> Step 3 [ How can we determine valid positions for a single rook such that the placement remains peaceful even after a 180-degree rotation of the board? ] 结果符合要求                
                
                # 依赖性分析
                relations_test = construct_dependencies_without_traversal(question, steps, config)  # query LLM回答所有的依赖
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
                        
                        # question 问题字符串
                        # 交待解决任务
                        sys_q = f"""There is a math_problem. I need you to solve it and give an answer.
Here is the problem:\n{question}

I have broken this math problem down into several smaller problems. I will assign you sub-problems one by one, and provide the results of the previous sub-problems as a reference for your reasoning.
Please solve the problem and respond according to mathematical logic.
        """  # 系统任务信息
                        
                        if len(answerDict)>0:
                            answersSoFar = f"""\nSo far, the answers to the resolved sub-problems are as follows: The format is Sub-problem-Id: xxx; Sub-problem: xxx; Answer: xxx."""
                            for key, value in answerDict.items():
                                answersSoFar += f"""\nSub-problem-Id: {key}; Sub-problem: {answerDict[key]['subtask']}; Answer: {answerDict[key]['answer']}."""
                            
                            predecessors = search_Predecessors(int_edges, number)
                            intersection = set(answerDict.keys()).intersection(set(predecessors))
                            count = len(intersection)
                            if count>0:
                                answersSoFar += f"""\nAmong them, sub-problems {predecessors} are directly related to this sub-problem, so please pay special attention to them."""
                        
                        
                        subask = f"""\nThe sub-problem to solve now is xxx: {subtask}
Based on the information above, please provide a concise and clear answer"""

                        if len(answerDict)>0:
                            query = answersSoFar+subask
                        else:
                            query = subask

                        Q = [{'role':'system', 'content':sys_q},
                            {'role':'user', 'content':query},]
                        
                        # 解决子任务的问答,其实这一步可以使用GPT-3.5来实现
                        # 在最后的总结推理阶段,使用GPT-4来实现.
                        # 这部分确实也不需要有规定的输出格式
                        result = askChatGPT(Q, model=config['subtask_MODEL'], temperature=1, max_tokens=300)
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
                Q.append({'role':'user', 'content':f"""Now that all the sub-problems have been solved, so what is the final answer?
Please give the final answer without any additional explanation or clarification."""})
                finalResult = askChatGPT(Q, model=config["finalSummarize_MODEL"], temperature=1)
                # print('图上推理 done')
                
                # print('\n\nFinalResult:')
                # print(finalResult)
                
                # 让大语言模型来判断有没有回答正确
                judgeAnswer = {'role':'user', 'content':f"""Here is a math problem with a standard answer and a student's solution. Please help me determine if the student's solution is correct.
Problem: {question}

Standard answer: {gold_answer}

Answer: {finalResult}

If the student's answer is correct, just output True; otherwise, just output False.
No explanation is required.
"""}
                Q_judge = [judgeAnswer]
                ifcorrect = askChatGPT(Q_judge, model=config["judgeCorrect_MODEL"], temperature=1)  # 要么是True, 要么是False
                
                if 'True' in ifcorrect:
                    success_Q += 1
                    print('correct')
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功
                elif 'False' in ifcorrect:
                    unsuccess_Q += 1
                    print('error')
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功                       
                
                # debug阶段,只测试一次
                break
            
            
            # except:
            #     attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
            #     print(f"error: {attempts};  taskid: {question_id}")  # 生成过程出错了
        
        # if attempts == 3:
        #     error_Q += 1
        #     print('run error 3+')
          
    print(f'correct_Q: {success_Q}')
    print(f'error_Q: {error_Q}')
    
    
    