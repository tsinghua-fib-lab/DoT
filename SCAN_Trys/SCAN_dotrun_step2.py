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
from SCAN_utils import *
from tqdm import tqdm
from utils import *
sys.path.append('../')

# client定义需要满足如下调用方式: client.chat.completions.create(model,messages = messages), 详见askLLM函数
openaiClient = setOpenAi(keyid = 0)
llamaClient = setLocal()
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "final_version-step2"

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
    
    f = open('TmpRes/step2In_SCAN_last.json', 'r')
    content = f.read()
    middleRes = json.loads(content) 
    MAX_TRY = 5
    elapsed_time = 0
    
    # 选择问题
    for question_id in tqdm(question_ids):
        
        question = tasks[question_id]
        
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('instruction content:')
        logger.info(question)

        attempts = 0
        success = False
        while attempts < 3 and not success:  # 如果遇到格式错误
            try:
                start_time = time.time()
                steps, steps_dict, allo_model, depths, int_edges = middleRes[str(question_id)]['steps'], middleRes[str(question_id)]['steps_dict'], middleRes[str(question_id)]['allo_model'], middleRes[str(question_id)]['depths'], middleRes[str(question_id)]['int_edges']
                # 这些量都是需要加载的
                depths = {int(k): v for k, v in depths.items()}
                heights = list(depths.keys())
                MAXHeight = max(heights)
                answerDict = {}  # 只有已经做过回答的subtask才会被放到这里面来
                progress_bar = tqdm(total=len(steps))
                
                
                for i in range(MAXHeight):
                    subtasks = depths[i]
                    for subtaskid in subtasks:                
                        number = re.findall(r'\d+', subtaskid)
                        number = int(number[0]) if number else None
                        subtask = steps_dict[str(number)]
                        answer_MODEL = allo_model[number-1]
                       
                        
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

                actionSeq = sentenceRes2Actions(clients, finalResult, config)
                actionList = actionSeq.split()
                
                logger.info('answer: '+str(actionList))
                logger.info('gold: '+str(solutions[question_id]))
                end_time = time.time()
                
                # 理论上结果直接和真实结果对比一下,算个正确率即可
                if actionList == solutions[question_id]:
                    success_Q += 1
                    logger.info('correct')
                    success = True
                    duration_time = end_time - start_time
                    elapsed_time += duration_time
                else:
                    unsuccess_Q += 1
                    logger.info('error')
                    attempts += 1
            except:
                attempts += 1  # 如果在执行过程中报错中止,还有重做的机会
                print(f"error: {attempts};  taskid: {question_id}")  # 生成过程出错了
        
        if attempts == 3:
            error_Q += 1
            logger.info('run error 3+')
    
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
    
    
    