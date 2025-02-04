'''
以alpha-quantile概率作为标准来初始化模型分配
其实这样产出的就是一个确定性的结果
把所有的模型分配的结果以及初始的alpha概率给集合到一个结果中来
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

sys.path.append('C:\\Users\\Pluto\\Desktop\\TaDe')  # 添加路径方便import
from CHAMP_Trys.CHAMP_utils import *
from utils import *

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
openaiClient = setOpenAi(keyid = 0)
llamaClient = Groq(  # 这个是Groq调用llama的api接口
    api_key='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'
)
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "alpha-quantile初始化模型分配结果_1001"

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
    file_path = '../Task_Datasets/CHAMP/all_champ_p.json'
    N = 100
    with open(file_path, 'r', encoding='utf-8') as file:
        problems = json.load(file)

    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    question_ids = list(range(94, N))
    
    '''一组超参设置'''
    MAX_TRY = 2  # 错误尝试最大值
    alpha = 0.3
    threshold = 0.70
    thres5=[0.9, 0.8, 0.5, 0.4, 0.2]
    allResInfo = {}
    
    f = open('TmpRes/allResInfo.json', 'r')
    content = f.read()
    allResInfo = json.loads(content)
    
    
    
    
    '''开始模型分配'''
    for question_id in tqdm(question_ids):
        allResInfo[question_id] = {}
        # 一个问题的分解文件中有若干个分解答案,所以需要区分
        with open(f'Decomposed_Steps/{question_id}.json', 'r', encoding='utf-8') as f:
            pre_steps = json.load(f)
        
        question = problems[question_id]['problem_text']
        gold_answer = problems[question_id]['problem_answer']
        
        # 每个问题解2次        
        for solve_time in range(20):
            # 保存两次solution的分配结果.
            allResInfo[question_id][solve_time] = {}
            check_and_create_txt_file(f'ModelAllocation/Alpha/Allo_search/{question_id}_{solve_time}.txt')
            check_and_create_txt_file(f'ModelAllocation/Alpha/Allo_search_num/{question_id}_{solve_time}.txt')
            
            logger.info('\n\n\n')
            logger.info(f'number id: {question_id}')

            steps, steps_dict, depths, int_edges = pre_steps[solve_time]
            
            attempts = 0
            success = False
            while attempts < MAX_TRY and not success:  # 如果遇到格式错误
                try:
                    # 直接开始推理
                    heights = list(depths.keys())
                    MAXHeight = max(heights)
                    answerDict = {}  # 只有已经做过回答的subtask才会被放到这里面来
                    progress_bar = tqdm(total=len(steps))
                    
                    frac_models = {}
                    sub_probs = {}
                    for i in range(MAXHeight):
                        subtasks = depths[i]
                        for subtaskid in subtasks: 
                            number = re.findall(r'\d+', subtaskid)
                            number = int(number[0]) if number else None
                            subtask = steps_dict[str(number)]
                            
                            # 用gpt-3.5-turbo进行试回答
                            answer_MODEL = config["alpha-quantile-base"]  
                            
                            result, prob_values, Q = solve_Sub_Question(clients, question, answerDict, int_edges, number, subtask, tokens_path, answer_MODEL, alpha)
                            # prob_values 是从小到大排列的,而且已经在e上求了指数
                            frac = quantile(prob_values, alpha)  # 前30%小的位置
                            allo_frac = frac2model(frac, thres5)
                            if allo_frac!=config["alpha-quantile-base"]:
                                result, prob_values, Q = solve_Sub_Question(clients, question, answerDict, int_edges, number, subtask, tokens_path, allo_frac, alpha) 
                            answerDict[number] = {'subtask':subtask, 'answer':result}
                            frac_models[number] = allo_frac
                            sub_probs[number] = prob_values
                            progress_bar.update(1)
                                
                    progress_bar.close()
                    # print(frac_models)

                    # 已经问完了所有的subtask,最后问一次得到最终的答案
                    Q.append({'role':'assistant', 'content':result})
                    Q.append({'role':'user', 'content':f"""Now that all the sub-problems have been solved, so what is the final answer?
Please give the final answer without any additional explanation or clarification."""})
                    # finalResult = askChatGPT(Q, model=config["finalSummarize_MODEL"], temperature=1)
                    finalResult = askLLM(clients, Q, tokens_path=tokens_path, model=config['finalSummarize_MODEL'], temperature=1, max_tokens=300)
                    
                    # print('推理 all done')
                    # logger.info('finalResult:\n')
                    # logger.info(finalResult)
                    
                    # 让大语言模型来判断有没有回答正确
                    judgeAnswer = {'role':'user', 'content':f"""Here is a math problem with a standard answer and a student's solution. Please help me determine if the student's solution is correct.
Problem: {question}

Standard answer: {gold_answer}

Answer: {finalResult}

If the student's answer is correct, just output True; otherwise, just output False.
No explanation is required.
"""}
                    Q_judge = [judgeAnswer]
                    ifcorrect = askLLM(clients, Q_judge, tokens_path=tokens_path, model=config['judgeCorrect_MODEL'], temperature=1, max_tokens=300)
                    
                    if 'True' in ifcorrect:
                        success_Q += 1
                        logger.info('correct')
                        success = True  # 任务未受中断,完整地结束了,所以标记为成功
                        with open(f'ModelAllocation/Alpha/Allo_search/{question_id}_{solve_time}.txt', 'a') as f:
                            f.write(str(frac_models)+' True\n')
                        
                        num_allo_model = {k: model_mapping[v] for k, v in frac_models.items()}
                        num_allo_model = list(num_allo_model.values())
                        dict_str = str(num_allo_model)
                        # 将字符串写入到指定的 txt 文件中
                        with open(f'ModelAllocation/Alpha/Allo_search_num/{question_id}_{solve_time}.txt', 'a') as f:
                            f.write(dict_str+' True\n')
                        allResInfo[question_id][solve_time]['allo_str'] = frac_models
                        allResInfo[question_id][solve_time]['allo_int'] = num_allo_model
                        allResInfo[question_id][solve_time]['result'] = True
                        allResInfo[question_id][solve_time]['sub_probs'] = sub_probs
                    elif 'False' in ifcorrect:
                        unsuccess_Q += 1
                        logger.info('error')
                        success = True  # 任务未受中断,完整地结束了,所以标记为成功    
                        with open(f'ModelAllocation/Alpha/Allo_search/{question_id}_{solve_time}.txt', 'a') as f:
                            f.write(str(frac_models)+' False\n')       
                        num_allo_model = {k: model_mapping[v] for k, v in frac_models.items()}
                        num_allo_model = list(num_allo_model.values())
                        dict_str = str(num_allo_model)
                        # 将字符串写入到指定的 txt 文件中
                        with open(f'ModelAllocation/Alpha/Allo_search_num/{question_id}_{solve_time}.txt', 'a') as f:
                            f.write(dict_str+' False\n')  
                        allResInfo[question_id][solve_time]['allo_str'] = frac_models
                        allResInfo[question_id][solve_time]['allo_int'] = num_allo_model
                        allResInfo[question_id][solve_time]['result'] = False
                        allResInfo[question_id][solve_time]['sub_probs'] = sub_probs          
                except:
                    attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
                    logger.info(f"error: {attempts};  taskid: {question_id}")  # 生成过程出错了
            
            if attempts == MAX_TRY:     # 一律当做错误推理处理
                error_Q += 1
                logger.info(f'run error {MAX_TRY}+')
                with open(f'ModelAllocation/Alpha/Allo_search/{question_id}_{solve_time}.txt', 'a') as f:
                    f.write(str(frac_models)+' False\n')
                num_allo_model = {k: model_mapping[v] for k, v in frac_models.items()}
                num_allo_model = list(num_allo_model.values())
                dict_str = str(num_allo_model)
                # 将字符串写入到指定的 txt 文件中
                with open(f'ModelAllocation/Alpha/Allo_search_num/{question_id}_{solve_time}.txt', 'a') as f:
                    f.write(dict_str+' False\n')  
                allResInfo[question_id][solve_time]['allo_str'] = frac_models
                allResInfo[question_id][solve_time]['allo_int'] = num_allo_model
                allResInfo[question_id][solve_time]['result'] = False
                allResInfo[question_id][solve_time]['sub_probs'] = sub_probs   
                       
            # 每次
                      
            write_json_listoneline('TmpRes/allResInfo.json', allResInfo)
            
    end_time = time.time()
    # 计算运行时间
    elapsed_time = end_time - start_time
    hours, minutes, seconds = seconds_to_hms(elapsed_time)
    logger.info(f"solving 运行耗时: {hours}h, {minutes}min, {seconds}s")
    logger.info(f'\n{tokens_path}')
    logger.info(f'Correct_Q: {success_Q}')
    logger.info(f'False_Q: {unsuccess_Q}')
    logger.info(f'Error_Q: {error_Q}\n')

    # 计算token价格
    with open(tokens_path, 'r') as f:
        token_usage = json.load(f)
        # logger.info(json.dumps(token_usage, indent=4))
        total_tokens, total_cost = CountCost(token_usage)
        # 打印结果
        logger.info(f"Total Tokens: {total_tokens}")
        logger.info(f"Total Cost: ${total_cost:.2f}")
    
    
    