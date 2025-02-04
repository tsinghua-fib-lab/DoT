import os

import openai
from openai import OpenAI

task = 'math'

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None
    
    

def decompose_sql(client, question, type,model):   

    prompt_for_decompose = f"""I will now give you a math problem. The type of problem is {type}. Please help me translate this math problem into a series of step-by-step sub-problems.

1 examples are as follows:
Question: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice 30 years old, how old is Kody? 
Answer: 
To solve the question "How old is Kody?", we need to solve the following problems step by step:
1. How old is Mohamed now?
2. How old was Mohamed four years ago?
3. How old was Kody four years ago?
4. How old is Kody now?

Now the command is {question}, please decompose it into a series of easy-to-solve steps like the examples.
Answer Format: (Please write each broken-down question step on a separate line, starting with a number.)
To solve the question "xxx", we need to solve the following problems step by step:
1. sub-question 1
2. sub-question 2
3. sub-question 3
...
"""

    Q = {
        "role": "user",
        "content": prompt_for_decompose
    }
    # Query = Example+[Q]
    Query = [Q]
    result = askChatGPT(client, Query, model, temperature=1)
    return result


def convert_steps_to_format(decom_commands):
    # 截取“we need to know:”后的内容
    start_index = decom_commands.find("we need to solve the following problems step by step:") + len("we need to solve the following problems step by step:")
    subtasks_text = decom_commands[start_index:].strip()
    # 将每个子任务单独列出
    subtasks = subtasks_text.split('\n')
    subtasks = [task.strip().split('. ', 1)[-1] for task in subtasks]
    steps_dict = {index: value for index, value in enumerate(subtasks)}
    return subtasks, steps_dict



def askChatGPT(client, messages, model="gpt-3.5-turbo", temperature = float(1)):
    response = client.chat.completions.create(
                model = model,
                messages = messages,
                temperature = temperature,
            )
    n_input = response.usage.prompt_tokens
    n_output = response.usage.completion_tokens
    
    def addtoken_output(num):
        file_path = f"{os.getcwd()}/{model}_output_token.txt"
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
        file_path = f"{os.getcwd()}/{model}_input_token.txt"
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