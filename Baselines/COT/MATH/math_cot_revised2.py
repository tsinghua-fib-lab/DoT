import json
import logging
import os
from typing import List

import openai
# Assume these functions are defined in MATH_cot_utils.py
from MATH_cot_utils import (askChatGPT, convert_steps_to_format, decompose_sql,
                            last_boxed_only_string, remove_boxed, setOpenAi)
from tqdm import tqdm

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
GPT_MODEL = "gpt-4o"

from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://101.6.69.60:8000/v1",
)
GPT_MODEL = "llama3-8b-8192"

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    filename = f'math_tot_{GPT_MODEL}_logfile.log'
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger, filename

def process_question(question: str, problem_type: str, gold_answer: str, steps: List[str], logger):
    sys_q = f"""
    There is a math problem. I need you to solve it and give an answer.
    Here is the problem:\n{question}

    I have broken this math problem down into a series of smaller problems. I will assign you sub-problems one by one, and provide the results of the previous sub-problems as a reference for your reasoning.
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
        
        result = askChatGPT(Q, client, model=GPT_MODEL)
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
    final_result = askChatGPT(Q, client, model=GPT_MODEL)
    print(f"THE FINAL RESULT IS: {final_result}")

    # Evaluate answer
    judge_query = f"""
    Here is a math problem with a standard answer and a student's solution. Please help me determine if the student's solution is correct. If the numerical values are the same, then it is correct.
                   
    Problem: {question}
    Standard answer: {gold_answer}
    Answer: {final_result}

    If the student's answer is correct, just output True; otherwise, just output False.
    No explanation is required.
    """
    Q_judge = [{'role': 'user', 'content': judge_query}]
    is_correct = askChatGPT(Q_judge, client, model=GPT_MODEL)

    logger.info(f'final answer by model: {final_result}')
    logger.info(f'gold answer: {gold_answer}')
    logger.info('correct' if 'True' in is_correct else 'incorrect')

    return 'True' in is_correct

def main():
    logger, filename = setup_logger()

    file_path = 'C:\\Users\Pluto\Desktop\TaDe\Task_Datasets\MATH\math200.json'
    X = 200

    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    questions = []
    types = []
    gold_answers = []

    for item in dataset[:X]:
        questions.append(item['problem'])
        types.append(item['type'])
        gold_answers.append(remove_boxed(last_boxed_only_string(item['solution'])))

    success_Q, unsuccess_Q, error_Q = 0, 0, 0

    for question_id in tqdm(range(len(questions))):
        question = questions[question_id]
        problem_type = types[question_id]
        gold_answer = gold_answers[question_id]

        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('instruction content:')
        logger.info(question)

        try:
            decompose_steps = decompose_sql(client, question, problem_type, GPT_MODEL)
            steps, _ = convert_steps_to_format(decompose_steps)
            
            if process_question(question, problem_type, gold_answer, steps, logger):
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