import json
import logging
import os
import sys

from CHAMP_utils import *
from tqdm import tqdm

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
client = setOpenAi(keyid = 1)

GPT_MODEL = "gpt-4o"

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    filename = f'champ_cot_{GPT_MODEL}_logfile.log'
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger, filename

if __name__ == '__main__':
    logger, filename = setup_logger()

    file_path = 'C:\\Users\Pluto\Desktop\TaDe\Task_Datasets\CHAMP\\all_champ_p.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    
    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    N = 200  # 跑200个问题
    question_ids = range(N)

    for question_id in tqdm(question_ids):
        question = dataset[question_id]['problem_text']
        type = split_string_by_(dataset[question_id]['problem_identifier'])
        gold_answer = dataset[question_id]['problem_answer']

        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('instruction content:')
        logger.info(question)

        decompose_steps = decompose_sql(client, question, type, GPT_MODEL)
        steps, steps_dict = convert_steps_to_format(decompose_steps)
        num_steps = len(steps)
        logger.info(f'total number of steps: {len(steps)}')
        
        solution = []

        sys_q = f"""There is a math problem. I need you to solve it and give an answer.
Here is the problem:\n{question}

I have broken this math problem down into a series of smaller problems. I will assign you sub-problems one by one, and provide the results of the previous sub-problems as a reference for your reasoning.
Please solve the problem and respond according to mathematical logic.
"""  

        for i in range(num_steps):
            subtask = steps[i]
            subask = f"""\nThe sub-problem to solve now is: {subtask}
Based on the information above, please provide a concise and clear answer"""
            
            if len(solution) == 0:
                query = subask
                Q = [{'role':'system', 'content':sys_q},
                     {'role':'user', 'content':query}]
            else:
                answersSoFar = f"""\nSo far, the answers to the preceding sub-problems are as follows: The format is Sub-problem-Id: xxx; Sub-problem: xxx; Answer: xxx."""
                for index, value in enumerate(solution):
                    answersSoFar += f"""\nSub-problem-Id: {index}; Sub-problem: {steps[index]}; Answer: {value}."""
                query = answersSoFar + subask
                Q = [{'role':'system', 'content':sys_q},
                     {'role':'user', 'content':query}]

            result = askChatGPT(Q, client, model=GPT_MODEL, temperature=0.6)
            solution.append(result)

        user_q = f"""There is a math problem:\n{question}

        I have broken this math problem down into a series of smaller problems and each sub-problem is solved.
        The sub-problems and their corresponding answers are as follows. (Format: Sub-problem-Id: xxx; Sub-problem: xxx; Answer: xxx.)"""

        for index, value in enumerate(solution):
            user_q += f"""\nSub-problem-Id: {index}; Sub-problem: {steps[index]}; Answer: {value}."""

        Q = [{'role':'system', 'content':sys_q},
             {'role':'user', 'content':user_q},
             {'role':'user', 'content':f"""Now that all the sub-problems have been solved, what is the final answer?
        Please give the final answer without any additional explanation or clarification."""}]

        finalResult = askChatGPT(Q, client, model=GPT_MODEL, temperature=1)
        print("THE FINAL RESULT IS: ", finalResult)

        judgeAnswer = {'role':'user', 'content':f"""Here is a math problem with a standard answer and a student's solution. Please help me determine if the student's solution is correct.
        Problem: {question}

        Standard answer: {gold_answer}

        Answer: {finalResult}

        If the student's answer is correct, just output True; otherwise, just output False.
        No explanation is required.
        """}

        Q_judge = [judgeAnswer]
        ifcorrect = askChatGPT(Q_judge, client, model=GPT_MODEL, temperature=1)

        logger.info(f'final answer: {finalResult}')
        logger.info(f'gold answer: {gold_answer}')

        if 'True' in ifcorrect:
            success_Q += 1
            logger.info('correct')
        elif 'False' in ifcorrect:
            unsuccess_Q += 1
            logger.info('incorrect')
        else:
            error_Q += 1
            logger.info('error in judgment')

        logger.info(f'accu: {success_Q/(success_Q+error_Q+unsuccess_Q)}')
            
    logger.info(f'correct_Q: {success_Q}')
    logger.info(f'error_Q: {error_Q}')
    logger.info(f'incorrect_Q: {unsuccess_Q}')
    logger.info(f'sum_Q: {success_Q+error_Q+unsuccess_Q}')
    
    
    