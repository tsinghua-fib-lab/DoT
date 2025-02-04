import json
import logging
import os
from typing import List

import openai
# Assume these functions are defined in DROP_3_utils.py
from DROP_3_utils import (askChatGPT, convert_steps_to_format, decompose_sql,
                          setOpenAi)
from openai import OpenAI
from tqdm import tqdm

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
gptclient = setOpenAi(keyid = 0)

llamaclient = OpenAI(
    api_key="EMPTY",
    base_url="http://101.6.69.60:8000/v1",
)
GPT_MODEL = "gpt-4o"

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    filename = f'drop_tot_{GPT_MODEL}_logfile.log'
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger, filename

def process_question(question: str, passage: str, gold_answer: str, steps: List[str], logger, client, model):
    sys_q = f"""
    There is an arithmetic reasoning problem. I need you to solve it and give an answer.

    Here is the context of the problem:\n{passage} 

    Here is the problem:\n{question}

    I have broken this problem down into a series of smaller problems. I will assign you sub-problems one by one, and provide the results of the previous sub-problems as a reference for your reasoning.
    Please solve the problem and respond according to mathematical logic.
    """

    solution = []
    for i, subtask in enumerate(steps):
        subask = f"\nThe sub-problem to solve now is: {subtask}\nBased on the information above, please provide a concise and clear answer"
        
        answersSoFar = "\nSo far, the answers to the preceding sub-problems are as follows: The format is Sub-problem-Id: xxx; Sub-problem: xxx; Answer: xxx."
        for index, value in enumerate(solution):
            answersSoFar += f"\nSub-problem-Id: {index}; Sub-problem: {steps[index]}; Answer: {value}."
        
        query = answersSoFar + subask if solution else subask
        Q = [{'role': 'system', 'content': sys_q},
             {'role': 'user', 'content': query}]
        
        result = askChatGPT(Q, client, model=model)
        solution.append(result)

    # Get final answer
    final_query = f"""
    Now that all the sub-problems have been solved, what is the final answer?
    Please give the final answer without any additional explanation or clarification.

    Problem: {question}

    Sub-problems and their answers:
    """
    for index, value in enumerate(solution):
        final_query += f"\nSub-problem-Id: {index}; Sub-problem: {steps[index]}; Answer: {value}."

    Q = [{'role': 'system', 'content': sys_q},
         {'role': 'user', 'content': final_query}]
    final_result = askChatGPT(Q, client, model=model)

    # Evaluate answer
    judge_query = f"""
    Here is a math problem with a standard answer and a student's solution. Please help me determine if the student's solution is correct. If the numerical answers are the same, return true.
                   
    Problem: {question}
    Standard answer: {gold_answer}
    Answer: {final_result}

    If the student's answer is correct, just output True; otherwise, just output False.
    No explanation is required.
    """
    Q_judge = [{'role': 'user', 'content': judge_query}]
    is_correct = askChatGPT(Q_judge, client, model=model)

    logger.info(f'final answer: {final_result}')
    logger.info(f'gold answer: {gold_answer}')
    logger.info('correct' if 'True' in is_correct else 'incorrect')

    return 'True' in is_correct

def main():
    logger, filename = setup_logger()

    with open('C:\\Users\Pluto\Desktop\TaDe\Task_Datasets\DROP\\all_drop_p.json', 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    success_Q, unsuccess_Q, error_Q = 0, 0, 0
    N = 200

    for question_id in tqdm(range(N)):
        question = dataset[question_id]['question']
        passage = dataset[question_id]['passage']
        gold_answer = dataset[question_id]['answer']

        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('instruction content:')
        logger.info(question)
        
        akakakq = f"""There is an arithmetic reasoning problem. 
Here is the context of the problem:\n{passage} 
Here is the question:\n{question}
        
I need you to assess whether this question is simple or difficult.
Answer format: If you think it is hard. Answer 'hard'. If you think it is easy. Answer 'easy'.
No explanation is needed."""
        Q = [{'role':'user', 'content':akakakq}]
        result = askChatGPT(Q, gptclient, model="gpt-4o", temperature=0.6)
        print(result)
        if 'hard' in result:
            client  = gptclient
            GPT_MODEL = "gpt-4o"
            print("use gpt-4o")
        else:
            client = llamaclient
            GPT_MODEL = "llama3-8b-8192"
            print("use llama3-8b")

        try:
            decompose_steps = decompose_sql(question, client, 'type', GPT_MODEL)
            steps, _ = convert_steps_to_format(decompose_steps)
            
            if process_question(question, passage, gold_answer, steps, logger, client, GPT_MODEL):
                success_Q += 1
            else:
                unsuccess_Q += 1
        except Exception as e:
            error_Q += 1
            logger.info(f'run error: {str(e)}')

        logger.info(f'acc_Q: {success_Q/(success_Q+error_Q+unsuccess_Q)}')

    logger.info(f'correct_Q: {success_Q}')
    logger.info(f'error_Q: {error_Q}')
    logger.info(f'incorrect_Q: {unsuccess_Q}')
    logger.info(f'sum_Q: {success_Q+error_Q+unsuccess_Q}')

if __name__ == '__main__':
    main()