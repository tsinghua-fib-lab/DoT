import json
import logging
from typing import List

import openai
# Assume these functions are defined in P3_cot_utils.py
from P3_cot_utils import *
from tqdm import tqdm

GPT_MODEL = "gpt-4o"
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
client = setOpenAi(keyid = 0)

def setup_logger(model):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    filename = f'p3_tot_{model}_logfile.log'
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger, filename

def process_question(question: str, ans_type: str, logger, model):
    sys_q = f"""
    There is a programming puzzle. Your final goal is to find an input for the function that will return True.
    
    Here is the problem:\n{question}

    I have broken this puzzle down into a series of smaller problems. I will assign you sub-problems one by one, and provide the results of the previous sub-problems as a reference for your reasoning.
    
    Please solve the problem and respond according to the logic and results of the previous sub-problems.
    """

    decompose_steps = decompose_sql(client, question, model=model)
    steps, _ = convert_steps_to_format(decompose_steps)

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
    Only give the input in the format of a string and just give the answer without any additional explanation or clarification, no prefix or suffix.

    For example if the input should be x = 5, then you should only give the answer as 5 and not x = 5.
    For example, if the input is list = [1,2,3], then you should only give the answer as [1,2,3] and not list = [1,2,3].

    Problem: {question}

    Sub-problems and their answers:
    """
    for index, value in enumerate(solution):
        final_query += f"\nSub-problem-Id: {index}; Sub-problem: {steps[index]}; Answer: {value}."

    Q = [{'role': 'system', 'content': sys_q},
         {'role': 'user', 'content': final_query}]
    final_result = askChatGPT(Q, client, model=model)

    logger.info(f'final answer: {final_result}')
    logger.info(f'final answer type: {ans_type}')

    try:
        # Create a new local namespace for each question
        local_namespace = {}
        
        # Execute the question to define the sat function in the local namespace
        exec(question, globals(), local_namespace)
        
        converted_result = convert_to_type(ans_type, final_result)
        result = local_namespace['sat'](converted_result)
        return result
    except Exception as e:
        logger.error(f"An error occurred in final answer evaluation: {e}")
        return False

def main(model="gpt-4-turbo"):
    logger, filename = setup_logger(model)
    setOpenAi(keyid=0)

    with open("C:\\Users\Pluto\Desktop\TaDe\Task_Datasets\ProgramPuzzle\puzzles.json", "r") as f:
        puzzles = json.load(f)

    success_Q, unsuccess_Q, error_Q = 0, 0, 0
    N = 200

    for question_id in tqdm(range(N)):
        question = puzzles[question_id]["sat"]
        ans_type = puzzles[question_id]['ans_type']

        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('instruction content:')
        logger.info(question)

        try:
            if process_question(question, ans_type, logger, model):
                success_Q += 1
                logger.info('correct')
            else:
                unsuccess_Q += 1
                logger.info('incorrect')
        except Exception as e:
            error_Q += 1
            logger.error(f'run error: {str(e)}')

        logger.info(f'acc_Q: {success_Q/(success_Q+error_Q+unsuccess_Q)}')

    logger.info(f'correct_Q: {success_Q}')
    logger.info(f'error_Q: {error_Q}')
    logger.info(f'incorrect_Q: {unsuccess_Q}')
    logger.info(f'sum_Q: {success_Q+error_Q+unsuccess_Q}')

if __name__ == '__main__':
    main(GPT_MODEL)  