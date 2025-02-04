import openai
import os
import logging
import re
GPT_MODEL = "gpt-4-turbo-preview"
# from main import GPT_MODEL


def split_instruction(instruction):
    parts = instruction.split('OUT:')
    question = parts[0].replace('IN:', '').strip()
    answer = parts[1].strip()
    return question, answer


def setup_logger():
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the global logging level to INFO
    
    # Define the format of log messages
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Log to a file
    filename = 'scan_cot_gpt4turbo.log'  # You may want to customize the filename
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Log to console (optional, for debugging purposes)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger, filename

def run(question):
    prompt_cot = """ 
    Instruction:
    Your task is to process a series of command instructions and generate the appropriate sequence of commands.
    
    Here are some examples:
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

    
    Pay attention to the use of parentheses.
    """
    
    query_sys = {
        "role":"system",
        "content": prompt_cot
    }
    
    query_user = {
        "role":"user",
        "content": f"""Let's think Step by Step, solve this question: {question}
        The output should be a list of commands in a string. For example: "RUN" * 2 + "TURN_LEFT" + "LOOK" * 3.
        """
    }
    
    query = [query_sys, query_user]
    
    result = askChatGPT(query, model=GPT_MODEL,temperature=1)
    
    #print("******************************** sanity check ********************************")
    #print(result)
    
    # 做mapping 得到一个式子
    
    queyr_assistant = {
        "role":"assistant",
        "content": result
    }
    
    query.append(queyr_assistant)
    
    q_format = {
        "role":"user",
        "content": """
        What is the output? 
        Make sure only give the answer. Don't include any other information.
        If the answer is "RUN" * 2 + "TURN_LEFT" + "LOOK" * 3, you should only write 'RUN' * 2 + 'TURN_LEFT' + 'LOOK' * 3.
        """
    }
    
    query.append(q_format)
    
    ## 再问一次确认答案
    
    finalanswer = askChatGPT(query, model=GPT_MODEL,temperature=1)
    
   # print("******************************** final answer ********************************")
   # print(finalanswer)
    
    answer_assistant = {
        "role":"assistant",
        "content": finalanswer
    }
    
    query.append(answer_assistant)
    
    ## 最后把这个公式给转化成一系列的命令
    
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
    
    Q_change = {"role": "user", "content": f"""The pseudo action sequence to be converted is as follows: {finalanswer} Please change it to the action sequences.
Please JUST answer the result."""}
    
    Q_now = [rewrite_system, Q_change]
    
    actions = askChatGPT(Q_now, model=GPT_MODEL, temperature = 1)

    # 删除数字
    text_no_numbers_and_brackets = re.sub(r'[\d\(\)]', '', actions)
    # 替换 TURN 后面的空格为下划线
    text_formatted = re.sub(r'TURN\s+', 'TURN_', text_no_numbers_and_brackets)
    # 去掉多余的空格
    text_formatted = re.sub(r'\s+', ' ', text_formatted).strip()
    
    return text_formatted

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
    file_path = f"{os.getcwd()}/scan_cot_{GPT_MODEL}_output_token.txt"
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
    file_path = f"{os.getcwd()}/scan_cot_{GPT_MODEL}_input_token.txt"
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