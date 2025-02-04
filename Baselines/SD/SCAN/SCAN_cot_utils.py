import os
import re

import openai
from openai import OpenAI

task = "scan"



def split_instruction(instruction):
    parts = instruction.split('OUT:')
    question = parts[0].replace('IN:', '').strip()
    answer = parts[1].strip()
    return question, answer

def decompose_sql(question,client, model):   

    prompt_for_decompose = f"""
I will give you a piece of natural language command. I need you to decompose it to smaller commands.

8 examples are as follows:

Command: "look right after look twice"d by: "look right", "look twice".

Command: "jump opposite right thrice and walk"
Result of decomposition: "jump opposite right thrice" can be solved by: "jump opposite right", "jump opposite right thrice". "walk" can be solved by: "walk". So, "jump opposite right thrice and walk" can finally be solved by: "jump opposite right", "jump opposite right thrice", "walk".

Command: "run left twice and run right"
Result of decomposition: "run left twice" can be solved by: "run left", "run left twice". "run right" can be solved by "run right". So, "run left twice and run right" can finally be solved by: "run left", "run left twice", "run right".

Command: "run opposite right"
Result of decomposition: "look right after look twice" can be solve
Result of decomposition: "run opposite right" can finally be solved by "run opposite right".

Command: "look opposite right thrice after walk"
Result of decomposition: "look opposite right thrice" can be solved by: "look opposite right", "look opposite right thrice". "walk" can be solved by "walk". So, "look opposite right thrice after walk" can finally be solved by: "look opposite right", "look opposite right thrice", "walk".

Command: "jump around right"
Result of decomposition: "jump around right" can be solved by: "jump right", "jump around right". So, "jump around right" can finally be solved by: "jump right", "jump around right".

Command: "look around right thrice and walk"
Result of decomposition: "look around right thrice" can be solved by: "look right", "look around right", "look around right thrice". "walk" can be solved by "walk". So, "look around right thrice and walk" can finally be solved by: "look right", "look around right", "look around right thrice", "walk".

Command: "turn right after run right thrice"
Result of decomposition: "turn right" can be solved by: "turn right". "run right thrice" can be solved by: "run right", "run right thrice". So, "turn right after run right thrice" can finally be solved by: "turn right", "run right", "run right thrice".

Now the command is {question}, please decompose it into smaller commands like the examples.
Answer Format: xxx can be solved by: xxx. xxx can be solved by xxx. ... So, xxx can finally be solved by: "subcommand_0", "subcommand_1",...
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
    # 正则表达式匹配 can be solved by: 之后的引号中的内容
    pattern = r'can finally be solved by:\s*("[^"]*"(?:,\s*"[^"]*")*)'

    # 查找匹配项
    match = re.search(pattern, decom_commands)

    # 提取匹配内容并转换为列表
    if match:
        # 提取逗号分隔的引号中的内容
        commands_decomposed = re.findall(r'"([^"]*)"', match.group(1))
        steps_dict = {index + 1: value for index, value in enumerate(commands_decomposed)}
        return commands_decomposed, steps_dict
    else:
        return False


def sentenceRes2Actions(client,sentence,model="gpt-3.5-turbo"):
    
    rewrite_system = {"role": "system", "content": f"""
    Now I have a pseudo action sequence expression with parentheses and multiplication. I need you to help me convert this into a sequence of actions without an operator sign.
    6 examples are as follows:    
        
    Q: "JUMP" * 3
    Rewrite: "JUMP" * 3
    A: 1 JUMP 2 JUMP 3 JUMP

    Q: "RUN" * 4 * 2
    Rewrite: "RUN" * 8
    A: 1 RUN 2 RUN 3 RUN 4 RUN 5 RUN 6 RUN 7 RUN 8 RUN

    Q: "TURN RIGHT" + "WALK"
    Rewrite: "TURN RIGHT" + "WALK"
    A: TURN RIGHT WALK

    Q: ("TURN LEFT" + "LOOK") * 2 + "TURN LEFT" + "LOOK"
    Rewrite: ("TURN LEFT" + "LOOK") * 2 + "TURN LEFT" + "LOOK"
    A: 1 (TURN LEFT LOOK) 2 (TURN LEFT LOOK) TURN LEFT LOOK

    Q: ("TURN RIGHT" * 2 + "JUMP") * 4
    Rewrite: ("TURN RIGHT" * 2 + "JUMP") * 4
    A: 1 (1 TURN RIGHT 2 TURN RIGHT JUMP) 2 (1 TURN RIGHT 2 TURN RIGHT JUMP) 3 (1 TURN RIGHT 2 TURN RIGHT JUMP) 4 (1 TURN RIGHT 2 TURN RIGHT JUMP)

    Q: "TURN LEFT" * 2 + ("TURN RIGHT" + "WALK") * 4 * 2
    Rewrite: "TURN LEFT" * 2 + ("TURN RIGHT" + "WALK") * 8
    A: 1 TURN LEFT 2 TURN LEFT 1 (TURN RIGHT WALK) 2 (TURN RIGHT WALK) 3 (TURN RIGHT WALK) 4 (TURN RIGHT WALK) 5 (TURN RIGHT WALK) 6 (TURN RIGHT WALK) 7 (TURN RIGHT WALK) 8 (TURN RIGHT WALK)
    """}

    Q_change = {"role": "user", "content": f"""The pseudo action sequence to be converted is as follows: {sentence} Please change it to the action sequences.
Please JUST answer the result."""}
    Q_now = [rewrite_system, Q_change]
    actions = askChatGPT(Q_now, client, model, temperature = 1)

    # 删除数字
    text_no_numbers_and_brackets = re.sub(r'[\d\(\)]', '', actions)
    # 替换 TURN 后面的空格为下划线
    text_formatted = re.sub(r'TURN\s+', 'TURN_', text_no_numbers_and_brackets)
    # 去掉多余的空格
    text_formatted = re.sub(r'\s+', ' ', text_formatted).strip()
    return text_formatted



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