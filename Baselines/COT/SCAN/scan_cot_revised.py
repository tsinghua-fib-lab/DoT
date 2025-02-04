import json
import logging
from typing import List

import openai
from SCAN_cot_utils import *
from tqdm import tqdm

GPT_MODEL = "gpt-3.5-turbo"
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
client = setOpenAi(keyid = 1)

def setup_logger(model):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    filename = f'scan_tot_{model}_logfile.log'
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger, filename


def process_question(question: str, gold_answer: List[str], logger, model):
    sys_q = f"""
    There is a natural language instruction representing a sequence of actions. I need you to translate this sentence from natural language into a standardized meta-action sequence.
    Here is the instruction:\n{question}

    I have broken this instruction down into some smaller instructions. I will assign you sub-instructions one by one, and provide the results of the previous sub-instructions as a reference for your reasoning.
    Please organize your reasoning according to the combination and progression of actions.

    For your reference, here are 13 examples for translation together with the corresponding explanations:
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
    """

    decompose_steps = decompose_sql(question, client, model=model)
    steps, _ = convert_steps_to_format(decompose_steps)

    solution = []
    for i, subtask in enumerate(steps):
        subask = f"\nThe sub-instruction to be converted now is: {subtask}\nBased on the information above, please provide a concise and clear answer"
        
        answersSoFar = "\nSo far, the answers to the preceding sub-instructions are as follows: The format is Sub-problem-Id: xxx; Sub-problem: xxx; Answer: xxx."
        for index, value in enumerate(solution):
            answersSoFar += f"\nSub-problem-Id: {index}; Sub-problem: {steps[index]}; Answer: {value}."
        
        query = answersSoFar + subask if solution else subask
        Q = [{'role': 'system', 'content': sys_q},
             {'role': 'user', 'content': query}]
        
        result = askChatGPT(Q, client, model=model)
        solution.append(result)

    # Get final answer
    final_query = f"""
    Now that all the sub-instructions have been solved, what is the final answer?
    Please give the final answer without any additional explanation or clarification.
    Try to concatenate some of the sub-problems answers to get the final answer.

    Problem: {question}

    Sub-problems and their answers:
    """
    for index, value in enumerate(solution):
        final_query += f"\nSub-problem-Id: {index}; Sub-problem: {steps[index]}; Answer: {value}."

    Q = [{'role': 'system', 'content': sys_q},
         {'role': 'user', 'content': final_query}]
    final_result = askChatGPT(Q, client, model=model)

    actionSeq = sentenceRes2Actions(client,final_result,GPT_MODEL)
    actionList = actionSeq.split()

    logger.info(f'final answer: {actionList}')
    logger.info(f'gold answer: {gold_answer}')

    return actionList == gold_answer

def main(model="gpt-4o"):
    logger, filename = setup_logger(model)
    setOpenAi(keyid=1)

    tasks = []
    solutions = []
    N = 200

    with open("C:\\Users\Pluto\Desktop\TaDe\Task_Datasets\SCAN\SCAN_all_tasks.txt", 'r', encoding='utf-8') as file:
        for line in file:
            if len(tasks) == N:
                break
            question, actions = split_instruction(line.strip())
            tasks.append(question)
            actions = [action.replace("I_", "") for action in actions.split()]
            solutions.append(actions)

    success_Q, unsuccess_Q, error_Q = 0, 0, 0

    for question_id in tqdm(range(len(tasks))):
        question = tasks[question_id]
        gold_answer = solutions[question_id]

        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('instruction content:')
        logger.info(question)

        try:
            if process_question(question, gold_answer, logger, model):
                success_Q += 1
                logger.info('correct')
            else:
                unsuccess_Q += 1
                logger.info('incorrect')
        except Exception as e:
            error_Q += 1
            logger.error(f'run error: {str(e)}')

        logger.info(f'accuracy: {success_Q/(success_Q+unsuccess_Q+error_Q)}')

    logger.info(f'correct_Q: {success_Q}')
    logger.info(f'error_Q: {error_Q}')
    logger.info(f'incorrect_Q: {unsuccess_Q}')
    logger.info(f'sum_Q: {success_Q+error_Q+unsuccess_Q}')

if __name__ == '__main__':
    main(GPT_MODEL)