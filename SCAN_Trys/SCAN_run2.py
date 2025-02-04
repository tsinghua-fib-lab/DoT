'''
run2: 根据回答的alpha-quantile来对模型进行升级
'''

# -*- coding: utf-8 -*-
import json
import os
import re
import sys
import time
from datetime import datetime

from groq import Groq
from SCAN_utils import *
from tqdm import tqdm

from utils import *

sys.path.append('C:\\Users\\Pluto\\Desktop\\TaDe')

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

    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    question_ids = list(range(N))
    
    MAX_TRY = 5  # 错误尝试最大值
    SUB_UPGRADE_UPPER = 2  # 每个子问题最多的LLM升级次数 
    update_subtask = 0  # 记录了一共有多少个子任务用到了升级
    update_times = 0  # 记录了LLM模型选择一共升级了多少次
    num_subtasks = 0  # 记录了一共分解出了多少个子任务
    alpha = 0.3
    threshold = 0.70
    
    # 选择问题
    for question_id in tqdm(question_ids):
        question = tasks[question_id]
        
        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('instruction content:')
        logger.info(question)

        attempts = 0
        success = False
        while attempts < MAX_TRY and not success:  # 如果遇到格式错误
            try:
                # 问题分解
                this_update_subtask = 0  # 记录了一共有多少个子任务需要处理
                this_update_times = 0  # 记录了LLM模型选择一共升级了多少次
                
                # 问题分解
                decompose_steps = decompose_sql(clients, question, config)
                # decompose_steps: "walk opposite right thrice after run opposite right" can be solved by: "run opposite right", "walk opposite right thrice".
                # print(decompose_steps)  # 基本没有问题
                
                # 分解后格式规范化
                steps, steps_dict = convert_steps_to_format(decompose_steps)
                # commands_decomposed: ['run opposite right', 'walk opposite right thrice']
                # print(steps_dict)
                
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
                        
                        # 先用base model回答一次check分位数
                        result, frac, Q = solve_Sub_Question(clients, answer_MODEL, question, answerDict, int_edges, number, subtask, tokens_path, alpha)
                        
                        while sub_upgrade_times < SUB_UPGRADE_UPPER:
                            if frac < threshold and answer_MODEL != 'gpt-4-turbo':
                                this_update_times += 1  # 记录了LLM模型选择一共升级了多少次
                                sub_upgrade_times += 1
                                ifUpdateLLM = True
                                
                                answer_MODEL = upGradeModel(answer_MODEL)
                                result, frac, Q = solve_Sub_Question(clients, answer_MODEL, question, answerDict, int_edges, number, subtask, tokens_path, alpha)      
                            else:
                                break
                            
                        answerDict[number] = {'subtask':subtask, 'answer':result}
                        progress_bar.update(1)
                        if ifUpdateLLM:
                            this_update_subtask += 1

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
                
                # print(actionList)
                # print(solutions[question_id])
                # print(actionList == solutions[question_id]) # list和list进行比较，比str比较稍微靠谱些
                

                # 理论上结果直接和真实结果对比一下,算个正确率即可
                if actionList == solutions[question_id]:
                    success_Q += 1
                    logger.info('correct')
                else:
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
    logger.info(f'correct_Q: {success_Q}')
    logger.info(f'False_Q: {unsuccess_Q}')
    logger.info(f'error_Q: {error_Q}')
    
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
    
    
    