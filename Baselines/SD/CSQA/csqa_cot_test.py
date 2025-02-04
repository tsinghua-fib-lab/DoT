import json
import logging
import os
import sys

from csqa_utils import *
from tqdm import tqdm

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
gptclient = setOpenAi(keyid = 1)

llamaclient = OpenAI(
    api_key="EMPTY",
    base_url="http://101.6.69.60:8001/v1",
)

GPT_MODEL = "gpt-4o"

def setup_logger():
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the global logging level to INFO
    
    # Define the format of log messages
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Log to a file
    filename = f'csqa_sd.log'  # You may want to customize the filename
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Log to console (optional, for debugging purposes)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger, filename

if __name__ == '__main__':
    logger, filename = setup_logger()

    dataset = []
    with open('C:\\Users\Pluto\Desktop\TaDe\Task_Datasets\CSQA\\train_rand_split.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    N = 200
    dataset = dataset[:N]
    
    success_Q = 0
    unsuccess_Q = 0
    error_Q = 0
    question_ids = range(len(dataset))

    for question_id in tqdm(question_ids):
        question = dataset[question_id]["question"]["stem"]
        choices = dataset[question_id]["question"]["choices"]
        gold_answer = dataset[question_id]["answerKey"]

        logger.info('\n\n\n')
        logger.info(f'number id: {question_id}')
        logger.info('instruction content:')
        logger.info(question)
        logger.info(choices)
        
        akakakq = f"""There is a common sense question:\n{question}
The choices are: {choices}
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
            # Problem decomposition
            decompose_steps = decompose_sql(client, question, choices,GPT_MODEL)
            steps, steps_dict = convert_steps_to_format(decompose_steps)
            num_steps = len(steps)
            logger.info(f'total number of steps: {num_steps}')

            solution = []

            sys_q = f"""
            There is a common sense question. I need you to solve it and give an answer.
            
            Here is the problem:\n{question}
            
            The choices are: {choices}

            I have broken this problem down into a series of smaller problems. I will assign you sub-problems one by one, and provide the results of the previous sub-problems as a reference for your reasoning.
            Please solve the problem and respond according to logic.
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

                result = askChatGPT(Q, client, model=GPT_MODEL, temperature=1)
                solution.append(result)

            # Final answer generation
            user_q = f"""There is a common sense question:\n{question}

            I have broken this problem down into a series of smaller problems and each sub-problem is solved.
            The sub-problems and their corresponding answers are as follows. (Format: Sub-problem-Id: xxx; Sub-problem: xxx; Answer: xxx.)"""

            for index, value in enumerate(solution):
                user_q += f"""\nSub-problem-Id: {index}; Sub-problem: {steps[index]}; Answer: {value}."""

            Q = [{'role':'system', 'content':sys_q},
                {'role':'user', 'content':user_q},
                {'role':'user', 'content':f"""Now that all the sub-problems have been solved, what is the final answer?
            Please Only give the letter of the correct answer, no need to provide the reason or the process or information. 
            For example: If the answer is A, please output 'A'. """}]

            finalResult = askChatGPT(Q, client, model=GPT_MODEL, temperature=1)

            logger.info(f'final answer: {finalResult}')
            logger.info(f'gold answer: {gold_answer}')

            if finalResult == gold_answer:
                success_Q += 1
                logger.info('correct')
            else:
                unsuccess_Q += 1
                logger.info('incorrect')

        except Exception as e:
            error_Q += 1
            logger.error(f"An error occurred: {e}")
            logger.info('run error')

    logger.info(f'correct_Q: {success_Q}')
    logger.info(f'error_Q: {error_Q}')
    logger.info(f'incorrect_Q: {unsuccess_Q}')
    logger.info(f'sum_Q: {success_Q+error_Q+unsuccess_Q}')