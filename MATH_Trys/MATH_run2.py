'''
以alpha-quantile概率作为标准
'''
'''
Here is a math word problem. I will first provide a description of the problem to set the context. Then, I will ask a specific question that requires you to use the information from the problem description, along with calculation and reasoning, to solve it.
'''

# -*- coding: utf-8 -*-
import json
import os
import re
import sys
import time
from datetime import datetime

from groq import Groq
from tqdm import tqdm

sys.path.append('C:\\Users\\Pluto\\Desktop\\TaDe')
from MATH_Trys.MATH_utils import *
from utils import *

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
openaiClient = setOpenAi(keyid = 0)
llamaClient = Groq(  # 这个是Groq调用llama的api接口
    api_key='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'
)
clients = {'gpt': openaiClient, 'llama': llamaClient}
aftername = "run2_alpha-quantile自动化任务升级"

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
    
    with open('MATH_config.json', 'r') as f:
        config = json.load(f)
    config['tokens_path'] = tokens_path
        
    file_path = '../Task_Datasets/MATH/math200.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        problems = json.load(file)

    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    N = 100
    question_ids = list(range(N))
    
    MAX_TRY = 5  # 错误尝试最大值
    SUB_UPGRADE_UPPER = 2  # 每个子问题最多的LLM升级次数 
    update_subtask = 0  # 记录了一共有多少个子任务用到了升级
    update_times = 0  # 记录了LLM模型选择一共升级了多少次
    num_subtasks = 0  # 记录了一共分解出了多少个子任务
    alpha = 0.3
    threshold = 0.75
    
    # 选择问题
    for question_id in tqdm(question_ids):
        
        question = problems[question_id]['problem']
        type = problems[question_id]['type']
        gold_answer = problems[question_id]['solution']
        
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('problem content:\n')
        logger.info(question)

        attempts = 0
        success = False
        while attempts < MAX_TRY and not success:  # 如果遇到格式错误
            try:
                # 问题分解
                this_update_subtask = 0  # 记录了一共有多少个子任务需要处理
                this_update_times = 0  # 记录了LLM模型选择一共升级了多少次
                
                # 问题分解
                decompose_steps = decompose_sql(clients, question, type, config)
                 
                # 分解后格式规范化
                steps, steps_dict = convert_steps_to_format(decompose_steps) 
                
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
                        ifUpdateLLM = False  # 对每个子任务归0     
                        sub_upgrade_times = 0
                                   
                        number = re.findall(r'\d+', subtaskid)
                        number = int(number[0]) if number else None
                        subtask = steps_dict[number]
                        answer_MODEL = config["alpha-quantile-base"]
                        
                        result, frac, Q = solve_Sub_Question(clients, question, answerDict, int_edges, number, subtask, tokens_path, answer_MODEL, alpha)
                        
                        while sub_upgrade_times < SUB_UPGRADE_UPPER:
                            if frac < threshold and answer_MODEL != 'gpt-4-turbo':
                                this_update_times += 1  # 记录了LLM模型选择一共升级了多少次
                                sub_upgrade_times += 1
                                ifUpdateLLM = True
                                
                                answer_MODEL = upGradeModel(answer_MODEL)
                                result, frac, Q = solve_Sub_Question(clients, question, answerDict, int_edges, number, subtask, tokens_path, answer_MODEL, alpha)    
                            else:
                                break
                        
                        answerDict[number] = {'subtask':subtask, 'answer':result}
                        progress_bar.update(1)
                        if ifUpdateLLM:
                            this_update_subtask += 1

                progress_bar.close()

                # 已经问完了所有的subtask,最后问一次得到最终的答案
                Q.append({'role':'assistant', 'content':result})
                Q.append({'role':'user', 'content':f"""Now that all the sub-problems have been solved, so what is the final answer?
Please give the final answer without any additional explanation or clarification."""})
                # finalResult = askChatGPT(Q, model=GPT_MODEL, temperature=1)
                finalResult = askLLM(clients, Q, tokens_path=tokens_path, model=config['finalSummarize_MODEL'], temperature=1, max_tokens=300)
                # print('图上推理 done')
                
                logger.info('finalResult: ')
                logger.info(finalResult)

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
                ifcorrect = askLLM(clients, Q_judge, tokens_path=tokens_path, model=config['judgeCorrect_MODEL'], temperature=1, max_tokens=300)
                
                
                if 'True' in ifcorrect:
                    success_Q += 1
                    logger.info('correct')
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功
                elif 'False' in ifcorrect:
                    unsuccess_Q += 1
                    logger.info('error')
                    success = True  # 任务未受中断,完整地结束了,所以标记为成功                       
                
                num_subtasks += len(steps)
                update_subtask += this_update_subtask  # 记录了一共有多少个子任务需要处理
                update_times += this_update_times  # 记录了LLM模型选择一共升级了多少次
            
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
    
    logger.info(f'所有问题分解出来的子任务总数: {num_subtasks}')
    logger.info(f'用到模型升级的子任务数量: {update_subtask}')
    logger.info(f'LLM一共升级了多少次: {update_times}\n')
    
    # 读取文件并打印结果以验证
    with open(tokens_path, 'r') as f:
        token_usage = json.load(f)
        # logger.info(json.dumps(token_usage, indent=4))
        total_tokens, total_cost = CountCost(token_usage)
        # 打印结果
        logger.info(f"Total Tokens: {total_tokens}")
        logger.info(f"Total Cost: ${total_cost:.2f}")
            
    
    