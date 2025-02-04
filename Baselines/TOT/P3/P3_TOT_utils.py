import os
import re
import sys

import openai
from openai import OpenAI

task = "P3"
#GPT_MODEL = "gpt-3.5-turbo"
#GPT_MODEL = "gpt-4o-mini"

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



def remove_quotes(s: str) -> str:
    # 去除两边的引号
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    elif s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    else:
        return s
    

def decompose_sql(client, question, model):    
    
    question_example = f"""
You will be provided with a Programming Puzzle. Your task is to find an input that will make the program return True.
Here is the puzzle: 
def sat(start: int, k=1, lower=93, seq=[-61, -46, 89, 93, -13, 14, -95, -74, -92, -38, -93, 64, -78, 3, 92, -10, -4, 43, 72, 12, 3, -3, -15, -96, 72, -71, -30, 53, 17, -87, 49, 17, -69, 78, 6, -77, -99, 91, 13, 9, 81, -55, 75, 48, -65, 18, -83, 10, -12, 88, 60, -72, -7, -49, -56, -76, 82, 18, 77, 52, -92, -88, 39, 13, -16, 82, 4, 44, -19, 54, 6, 55, 77, -38, -30, -55, -16]):
    return 0 <= start <= len(seq) - k and sum(seq[start:start + k]) >= lower
    
You have to decompose the puzzle into multiple steps. Carefully consider the granularity of each subtask to ensure that each one is executable.

Answer format: Please follow the format below strictly when answering. No explanation is required.
STEP1 [ step task 1 ]
STEP2 [ step task 2 ]
...
"""
    answer_example = r"""
Here are some examples of how to decompose a programming puzzle:

Question:
def sat(different: str, d={'CHIRATICHUHUQUYZYPYW': 'kopakyquotyhaquome', 'QUEBYTEXTEXUROBEK': 'tituxa', 'ZUVU': 'xupovutexti', 'NATEXTESYTUBUMY': 'ponusewaquufot', 'THUK': 'gyvy', 'CETEXTOFENAXIXANEKA': 'xyjytextecywykoquo', 'SEKAMIWEHYTHYTEXTUCU': 'jehu', 'H': 'quicyquohofowejivun', 'KYTEXTIBAXUTAV': 'nygutextin', 'LYQUA': 'biruji', 'tizenyry': 'xavyquukoc'}):
    return different in d and all(k.islower() != different.islower() for k in d if k != different)


Step1 [The string `different` must be a key in the dictionary `d`. ]
Step2 [List possible candidates for `different`]
Step3 [For all other keys in the dictionary `d` (excluding `different`), the case (lowercase or uppercase) of each key must differ from the case of `different`.]
Step4 [Check if `different` is a Key in `d`]
Step5 [Verify that `different` is a key in the dictionary `d`.]
Step6 [Verify the Condition for Each Key: - Ensure that for each key `k` in `d` (excluding `different`), the expression `k.islower() != different.islower()` holds true]
Step7 [Iterate the verification process for all the listed candidates]
Step8 [Select the candidate for `different` that satisfy the conditions]
Step9 [Generate a Python code that could produce the answer]
Step10 [Derive the output by understanding the function]


Question
def sat(li: List[int]):
	return len(li) == 10 and li.count(li[3]) == 2

Step1 [ Create a list of length 10 Initialize a list with 10 arbitrary elements, such as numbers or null values.]
Step2 [ Select an element to place in the fourth position of the list (index 3). Choose a simple integer or any other type of object as the value for this position.]
Step3 [ Ensure the chosen fourth element appears exactly twice in the list. Place the fourth element in another position within the list, ensuring both positions are distinct and that the element only appears twice.]
Step4 [ Fill the remaining positions in the list other than the two predetermined positions. Use values different from the fourth element to fill the rest of the list, ensuring these values do not inadvertently increase the occurrence of the fourth element.]
Step5 [Verify that the list meets the requirements of the sat(li) function.]
Step6 [ Check if the list length is 10. ]
Step7 [ Check if the fourth element appears exactly twice.]
Step8 [Generate a Python code that could produce the answer]
Step9 [Derive the output by understanding the function]


Question: 
def sat(certificates: List[int], nums=[99210055, 4171577125, 459354525, 1534026075, 4255533095, 2441396441, 155962261]):
    return all(pow(cert, n - 1, n) > 1 for cert, n in zip(certificates, nums)) and len(certificates) == len(nums)

Task Decompose:
Step1 [The expression `pow(cert, n - 1, n)` computes \( \text{cert}^{(n-1)} \mod n \)]
Step2 [For `pow(cert, n - 1, n)` to be greater than 1, `cert` should not be a trivial case like 1 or 0.]
Step3 [Create a list `certificates` of the same length as `nums`.]
Step4 [Each element in `certificates` should be a number that satisfies the condition for the corresponding element in `nums`.]
Step5 [Ensure that for each pair `(cert, n)`, the condition `pow(cert, n - 1, n) > 1` holds true.]
Step6 [Confirm that the length of `certificates` is the same as the length of `nums`.]
Step7 [Generate a Python code that could produce the answer]
Step8 [Derive the output by understanding the function]

"""

    Example = [
        {"role": "user", "content": question_example},
        {"role": "assistant", "content": answer_example}
    ]


    prompt_for_decompose = f"""
You will be provided with a Programming Puzzle. Your task is to find an input that will make the program return True.

Here is the puzzle: {question}

You have to decompose the puzzle into multiple steps. Carefully consider the granularity of each subtask to ensure that each one is executable.

You should not include any reasoning step in the subtasks, they should be executable steps only, not truely solving the puzzle.

You should not propose any step to write code, only manually executable steps.

Answer format: Please follow the format below strictly when answering. No explanation is required.
STEP1 [ step task 1 ]
STEP2 [ step task 2 ]
STEP3 [ step task 3 ]
......

"""
    Q = {
        "role": "user",
        "content": prompt_for_decompose
    }
    Query = Example+[Q]
    result = askChatGPT(client, Query, model=model, temperature=1)
    # client, messages, model='gpt-3.5-turbo', temperature = float(1)
    return result


def convert_steps_to_format(raw_steps):
    lines = raw_steps.strip().split('\n')
    steps_dict = {}
    steps = []
    for line in lines:
        if line.strip()  and 'STEP' in line: # 只处理非空行
            step_number = int(line.split(' ')[0][4:])  # 提取数值部分并转换为整数
            step_id = line.split(' ')[0]
            step_content = line[line.index('[') + 1 : line.rindex(']')]
            steps_dict[step_number] = step_content
            steps.append({"stepId": step_number, "step": step_content})
            
            
    # return steps_dict
    return steps, steps_dict


def askChatGPT(client, messages, model='gpt-3.5-turbo', temperature = float(1)):
    response = client.chat.completions.create(
                model = model,
                messages = messages,
                temperature = temperature,
            )
    n_input = response.usage.prompt_tokens
    n_output = response.usage.completion_tokens
    
    def addtoken_output(num):
        file_path = f"{os.getcwd()}/drop_tot_{model}_output_token.txt"
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
        file_path = f"{os.getcwd()}/drop_tot_{model}_input_token.txt"
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
    
    addtoken_input(n_input)
    addtoken_output(n_output)
    answer = response.choices[0].message.content
    return answer.strip()

def split_string_by_(string):
    return string.split('_')[1]




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
        api_key = ""
    client = OpenAI(api_key=api_key)
    return client

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