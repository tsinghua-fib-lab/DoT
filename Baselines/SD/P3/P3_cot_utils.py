import os
import re

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
    

def decompose_sql(client, question,model):    
    
    question_example = f"""
You will be provided with a Programming Puzzle. Your task is to find an input that will make the program return True.
    
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


STEP1 [The string `different` must be a key in the dictionary `d`. ]
STEP2 [List possible candidates for `different`]
STEP3 [For all other keys in the dictionary `d` (excluding `different`), the case (lowercase or uppercase) of each key must differ from the case of `different`.]
STEP4 [Check if `different` is a Key in `d`]
STEP5 [Verify that `different` is a key in the dictionary `d`.]
STEP6 [Verify the Condition for Each Key: - Ensure that for each key `k` in `d` (excluding `different`), the expression `k.islower() != different.islower()` holds true]
STEP7 [Iterate the verification process for all the listed candidates]
STEP8 [Select the candidate for `different` that satisfy the conditions]
STEP9 [Generate a Python code that could produce the answer]
STEP10 [Derive the output by understanding the function]


Question
def sat(li: List[int]):
	return len(li) == 10 and li.count(li[3]) == 2

STEPp1 [ Create a list of length 10 Initialize a list with 10 arbitrary elements, such as numbers or null values.]
STEP2 [ Select an element to place in the fourth position of the list (index 3). Choose a simple integer or any other type of object as the value for this position.]
STEP3 [ Ensure the chosen fourth element appears exactly twice in the list. Place the fourth element in another position within the list, ensuring both positions are distinct and that the element only appears twice.]
STEP4 [ Fill the remaining positions in the list other than the two predetermined positions. Use values different from the fourth element to fill the rest of the list, ensuring these values do not inadvertently increase the occurrence of the fourth element.]
STEP5 [Verify that the list meets the requirements of the sat(li) function.]
STEP6 [ Check if the list length is 10. ]
STEP7 [ Check if the fourth element appears exactly twice.]
STEP8 [Generate a Python code that could produce the answer]
STEP9 [Derive the output by understanding the function]


Question: 
def sat(certificates: List[int], nums=[99210055, 4171577125, 459354525, 1534026075, 4255533095, 2441396441, 155962261]):
    return all(pow(cert, n - 1, n) > 1 for cert, n in zip(certificates, nums)) and len(certificates) == len(nums)

Task Decompose:
STEP1 [The expression `pow(cert, n - 1, n)` computes \( \text{cert}^{(n-1)} \mod n \)]
STEP2 [For `pow(cert, n - 1, n)` to be greater than 1, `cert` should not be a trivial case like 1 or 0.]
STEP3 [Create a list `certificates` of the same length as `nums`.]
STEP4 [Each element in `certificates` should be a number that satisfies the condition for the corresponding element in `nums`.]
STEP5 [Ensure that for each pair `(cert, n)`, the condition `pow(cert, n - 1, n) > 1` holds true.]
STEP6 [Confirm that the length of `certificates` is the same as the length of `nums`.]
STEP7 [Generate a Python code that could produce the answer]
STEP8 [Derive the output by understanding the function]

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
    result = askChatGPT(Query, client, model, temperature=1)
    return result


def convert_steps_to_format(raw_steps):
    lines = raw_steps.strip().split('\n')
    steps_dict = {}
    steps = []
    for line in lines:
        if line.strip() and 'STEP' in line:  # 只处理非空行
            step_number = int(line.split(' ')[0][4:])  # 提取数值部分并转换为整数
            # step_id = line.split(' ')[0]
            step_content = line[line.index('[') + 1 : line.rindex(']')]
            steps_dict[step_number] = step_content
            steps.append({"stepId": step_number, "step": step_content})
            
            
    # return steps_dict
    return steps, steps_dict


def askChatGPT(messages, client, model="gpt-3.5-turbo", temperature = float(1)):
    response = client.chat.completions.create(
                model = model,
                messages = messages,
                temperature = temperature,
            )
    n_input = response.usage.prompt_tokens
    n_output = response.usage.completion_tokens
    
    def addtoken_output(num):
        file_path = f"{os.getcwd()}/{task}_tot_{model}_output_token.txt"
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
        file_path = f"{os.getcwd()}/{task}_tot_{model}_input_token.txt"
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