import os

import openai
from openai import OpenAI

task = "champ"

# from main import GPT_MODEL
def decompose_sql(client, question, type,model):   

    prompt_for_decompose = f"""

I will now give you a math problem. The type of problem is {type}. Please help me translate this math problem into a series of step-by-step sub-problems.

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
    result = askChatGPT(Query, client, model, temperature=1)
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


def split_string_by_(string):
    return string.split('_')[1]

def askChatGPT(messages, client, model, temperature = 0.6):
    
    response = client.chat.completions.create(
                model = model,
                messages = messages,
                temperature = temperature,
            )
    n_input = response.usage.prompt_tokens
    n_output = response.usage.completion_tokens
    
    def addtoken_output(num):
        file_path = f"{os.getcwd()}/champ_cot_{model}_output_token.txt"
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
        file_path = f"{os.getcwd()}/champ_cot_{model}_input_token.txt"
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
    if model == "gpt-4o":
        addtoken_input(n_input)
        addtoken_output(n_output)
    answer = response.choices[0].message.content
    return answer.strip()

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


def CountCost(token_usage):
    cost_per_1000_tokens = {
        "gpt-4-turbo": {"prompt": 10, "completion": 30},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
        "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
        "gpt-4": {"prompt": 30, "completion": 60},
        "gpt-4o": {"prompt": 2.5, "completion": 10}  # 确实是使用4o更加便宜一点
    }

    # 计算总token和总成本
    total_tokens = 0
    total_cost = 0

    for model, tokens in token_usage.items():
        prompt_tokens = tokens['prompt_tokens']
        completion_tokens = tokens['completion_tokens']
        total_tokens += prompt_tokens + completion_tokens
        total_cost += (prompt_tokens / 1000000) * cost_per_1000_tokens[model]["prompt"]
        total_cost += (completion_tokens / 1000000) * cost_per_1000_tokens[model]["completion"]
    
    return total_tokens, total_cost

if __name__ == '__main__':
    print(judgeNum(1,1))