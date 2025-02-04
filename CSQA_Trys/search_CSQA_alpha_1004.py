'''
MATH数学问题
'''
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
from MATH_Trys.MATH_utils import *
from utils import *

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
openaiClient = setOpenAi(keyid = 0)
# llamaClient = Groq(  # 这个是Groq调用llama的api接口
#     api_key='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'
# )
llamaClient = OpenAI(
    api_key="EMPTY",
    base_url="http://101.6.69.111:8000/v1",
)
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "LLM_allocation_search-1004"



def _baseReason():
    heights = list(depths.keys())
    heights = [int(item) for item in heights]
    MAXHeight = max(heights)
    answerDict = {}  # 只有已经做过回答的subtask才会被放到这里面来
    for i in range(MAXHeight+1):
        subtasks = depths[str(i)]
        for subtaskid in subtasks:                
            number = re.findall(r'\d+', subtaskid)
            number = int(number[0]) if number else None
            subtask = steps_dict[str(number)]
            answer_MODEL = 'gpt-3.5-turbo'
            
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
            
            result, probs = askLLM_withprob(clients, Q, tokens_path=tokens_path, model=answer_MODEL, temperature=1, max_tokens=300)
            prob_values = list(probs.values())
            answerDict[number] = {'subtask':subtask, 'answer':result, 'probs':prob_values}
    return answerDict

    



def _reason(allo_model, solve_time):
    logger.info(allo_model)  # 打印一下模型分配的结果
    attempts = 0
    success = False
    while attempts < MAX_TRY and not success:  # 如果遇到格式错误
        try:            
            heights = list(depths.keys())
            heights = [int(item) for item in heights]
            MAXHeight = max(heights)
            answerDict = {}  # 只有已经做过回答的subtask才会被放到这里面来
            for i in range(MAXHeight+1):
                subtasks = depths[str(i)]
                for subtaskid in subtasks:                
                    number = re.findall(r'\d+', subtaskid)
                    number = int(number[0]) if number else None
                    subtask = steps_dict[str(number)]
                    answer_MODEL = allo_model[number]
                    
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
                    answerDict[number] = {'subtask':subtask, 'answer':result}

            # 已经问完了所有的subtask,最后问一次得到最终的答案
            Q.append({'role':'assistant', 'content':result})
            Q.append({'role':'user', 'content':f"""Now that all the sub-questions have been solved, which answer do you ultimately choose?
Please provide only the letter of the option, without any additional explanation or description."""})
            # finalResult = askChatGPT(Q, model=GPT_MODEL, temperature=1)
            finalResult = askLLM(clients, Q, tokens_path=tokens_path, model=config['finalSummarize_MODEL'], temperature=1, max_tokens=300)
            # print('最终回答 done')
            
            # logger.info('finalResult: ')
            # logger.info(finalResult)
            finalResult = finalResult.strip()
            ifcorrect = finalResult == gold_answer
            
            if ifcorrect:
                logger.info('True')
                dict_str = str(allo_model)
                # 将字符串写入到指定的 txt 文件中
                with open(f'ModelAllocation/Alpha-search/{question_id}_{solve_time}.txt', 'a') as f:
                    f.write(dict_str+' True\n')
                success = True       
                return 'True'
            else:
                logger.info('False')
                dict_str = str(allo_model)
                # 将字符串写入到指定的 txt 文件中
                with open(f'ModelAllocation/Alpha-search/{question_id}_{solve_time}.txt', 'a') as f:
                    f.write(dict_str+' False\n')
                success = True       
                return 'False'   
        except:
            attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
            logger.info(f"error: {attempts};  taskid: {question_id}")  # 生成过程出错了
    
    if attempts == MAX_TRY:
        logger.info(f'run error {MAX_TRY}+')
        return 'error'
            
            
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
        
    # file_path = '../Task_Datasets/MATH/math200.json'
    file_path = '../Task_Datasets/CSQA/train_rand_split.jsonl'
    questions = []
    with open(file_path, 'r') as f:
        # 逐行读取文件内容
        for line in f:
            # 将每一行转换为JSON对象
            entry = json.loads(line)
            # 处理或存储JSON对象
            questions.append(entry)
    
    decom_ids = extract_numbers_from_filenames('Decomposed_Steps_1004')
    decom_ids.sort()

    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    MAX_TRY = 5
    DecomTime = 20
    
    # 选择问题
    for question_id in tqdm(decom_ids):
        if question_id==2:
            pass
        
        with open(f'Decomposed_Steps_1004/{question_id}.json', 'r', encoding='utf-8') as f:
            pre_steps = json.load(f)
            
        question = questions[question_id]['question']['stem']  # 应该把选项也放进问题里,作为问题的一部分.
        options = questions[question_id]['question']['choices']  # CSQA数据集是给出具体选项的
        gold_answer = questions[question_id]['answerKey']
        options_string = "; ".join([f"{item['label']}: {item['text']}" for item in options])
        
            
        # 每个问题解2次        
        for solve_time in range(DecomTime):
        
            exi = check_and_create_txt_file(f'ModelAllocation/Alpha-search/{question_id}_{solve_time}.txt')
            if not exi:
                logger.info('\n\n\n')
                logger.info(f'number id: {question_id}  solve time: {solve_time}')

                steps, steps_dict, depths, int_edges = pre_steps[solve_time]
                depthlen = sum(len(value) for value in depths.values())
                if len(steps) != depthlen:
                    depths = {str(i): [f'Step {i+1}'] for i in range(len(steps))}
                    int_edges = [[i, i+1] for i in range(1, len(steps))]
                
                subAnswerDict = _baseReason()
                alpha0 = []
                alpha04 = []
                alpha08 = []
                for i in range(1, len(steps)+1):
                    alpha0.append(quantile(subAnswerDict[i]['probs'], 0))  # frac = quantile(prob_values, alpha)
                    alpha04.append(quantile(subAnswerDict[i]['probs'], 0.4))
                    alpha08.append(quantile(subAnswerDict[i]['probs'], 0.8))

                sorted_list, sorted_indices = sort_with_indices(alpha04)  # 排序靠前的都是自信的回答,可都用小模型做
                allo_model = {sorted_indices[i]+1:'gpt-4-turbo' if sorted_list[i]<0.9 else 'llama3-8b' for i in range(len(sorted_indices))}
                cixu = [x + 1 for x in sorted_indices]
                judgement = _reason(allo_model, solve_time=solve_time)
                largeNum = sum(1 for x in alpha04 if x < 0.9) 
                smallNum = len(steps) - largeNum
                # 如果原本可以完成
                if judgement == 'True':
                    time_downgrading = 0
                    while judgement == 'True' and time_downgrading < largeNum: 
                        index = find_first_valid_key(cixu, allo_model)
                        allo_model[index] = 'llama3-8b'
                        judgement = _reason(allo_model, solve_time=solve_time)
                        time_downgrading += 1
                else:
                    time_upgrading = 0
                    while judgement == 'False' and time_upgrading < smallNum:
                        index = find_first_valid_key2(cixu, allo_model)
                        allo_model[index] = 'gpt-4-turbo'
                        judgement = _reason(allo_model, solve_time=solve_time)
                        time_upgrading += 1
    
    # 读取文件并打印结果以验证
    with open(tokens_path, 'r') as f:
        token_usage = json.load(f)
        # logger.info(json.dumps(token_usage, indent=4))
        total_tokens, total_cost = CountCost(token_usage)
        # 打印结果
        logger.info(f"Total Tokens: {total_tokens}")
        logger.info(f"Total Cost: ${total_cost:.2f}")
            
    
    