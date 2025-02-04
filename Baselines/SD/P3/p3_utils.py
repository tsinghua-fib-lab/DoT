import openai
import os
import logging
import re

GPT_MODEL = "gpt-4-turbo-preview"  # [gpt-4-turbo-preview]
#GPT_MODEL = "gpt-3.5-turbo"  # [gpt-3.5-turbo-preview]


def cot_solve(question):
    cot_prompt = """
    You will be given a programming puzzle.
    Your task is to find an input for the following function that will make the function return True. 
    Let's think step by step"""
    
    input_q = [
        {
            "role": "system",
            "content": cot_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]
    
    procedure = askChatGPT(input_q, model = GPT_MODEL, temperature=1)
    ans_procedure = {"role": "assistant", "content": procedure}
    input_q.append(ans_procedure)
    #final_q = {"role": "user", "content": "So, what is the input? You can either give the input or write a solution function in Python. Make sure the answer you return in the format 'def function_name(): return value'. Make sure NO explanation!"}
    final_q = {"role": "user", "content": """So, what is the input? Please only give the input in the format of a string and just give the answer without any additional explanation or clarification, no prefix or suffix.

    For example if the input should be x = 5, then you should only give the answer as 5 and not x = 5.
    For example, if the the input is list = [1,2,3], then you should only give the answer as [1,2,3] and not list = [1,2,3].
                """}
    input_q.append(final_q)
    final_answer = askChatGPT(input_q, model = GPT_MODEL, temperature=1)
    return final_answer

    

from typing import Any, List
type_dict = {
    'str': str,
    'int': int,
    'float': float,
    'bool': bool,
    'List[int]': List[int],
    'List[str]': List[str],
    'List[float]': List[float],
    'List[bool]': List[bool],
    'List[List[int]]': List[List[int]],
    'List[List[float]]': List[List[float]],
    'List[List[str]]': List[List[str]],
    'List[List[List[int]]]': List[List[List[int]]]
}
def convert_to_type(type_str: str, value_str: str) -> Any:
    """
    根据给定的类型字符串和待转换的字符串,将其转换为相应的Python类型
    """
    if type_str in type_dict:
        python_type = type_dict[type_str]
        if type_str.startswith('List'):
            # 处理列表类型
            return eval(value_str)
        else:
            # 处理基本类型
            return python_type(value_str)
    else:
        raise ValueError(f"不支持的类型: {type_str}")

def setup_logger():
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the global logging level to INFO
    
    # Define the format of log messages
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Log to a file
    filename = f'p3_cot_{GPT_MODEL}.log'  # You may want to customize the filename
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Log to console (optional, for debugging purposes)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger, filename

def askChatGPT(messages, model="gpt-3.5-turbo", temperature = float(1)):
    response = openai.ChatCompletion.create(
            model = model,
            messages = messages,
            temperature = temperature
        )
    n_input = response.usage.prompt_tokens
    n_output = response.usage.completion_tokens
    addtoken_input(n_input)
    addtoken_output(n_output)
    answer = response.choices[0].message["content"]
    return answer.strip()

def split_string_by_(string):
    return string.split('_')[1]

def addtoken_output(num):
    file_path = f"{os.getcwd()}/p3_cot_{GPT_MODEL}_output_token.txt"
    tokens_used = num

    # Initialize cumulative_tokens
    cumulative_tokens = 0
    
    # Try to read the existing token count from the file
    try:
        with open(file_path, "r") as file:
            file_content = file.read().strip()
            if file_content:  # Check if the content is not empty
                cumulative_tokens = int(file_content)
    except FileNotFoundError:
        # If the file does not exist, we start with a cumulative_tokens of 0
        pass

    # Update the cumulative token count
    cumulative_tokens += tokens_used

    # Write the updated count back to the file
    with open(file_path, "w") as file:
        file.write(str(cumulative_tokens))

    return cumulative_tokens

def addtoken_input(num):
    file_path = f"{os.getcwd()}/p3_cot_{GPT_MODEL}_input_token.txt"
    tokens_used = num

    # Initialize cumulative_tokens
    cumulative_tokens = 0
    
    # Try to read the existing token count from the file
    try:
        with open(file_path, "r") as file:
            file_content = file.read().strip()
            if file_content:  # Check if the content is not empty
                cumulative_tokens = int(file_content)
    except FileNotFoundError:
        # If the file does not exist, we start with a cumulative_tokens of 0
        pass

    # Update the cumulative token count
    cumulative_tokens += tokens_used

    # Write the updated count back to the file
    with open(file_path, "w") as file:
        file.write(str(cumulative_tokens))

    return cumulative_tokens



def addtoken(num):
    try:
        with open("tokens.txt", "r") as f:  # 打开文件
            data = f.read()  # 读取文件
            nownum = int(data)        
            
        if num == -1:
            nownum = 0
        else:
            nownum = nownum + num
        
        with open("tokens.txt","w+") as f:
            f.write(str(nownum))  # 自带文件关闭功能，不需要再写f.close()
    except:
        pass
    
def setOpenAi(keyid = 0):
    # put your key here
    if keyid == 0:
        openai.api_key = ""
    return 0

def printSeq(seq):
    for item in seq:
        print(item)

def judgeNum(num1, num2):
    #num1 = num1.replace(',', '')
    #num2 = num2.replace(',', '')
    #num1 = int(num1)
    #num2 = int(num2)
    return 1 if num1 == num2 else 0

def judgeString(str1, str2):
    return 1 if str1 == str2 else 0

if __name__ == '__main__':
    print(judgeNum(1,1))