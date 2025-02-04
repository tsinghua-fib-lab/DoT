import openai
import os
# from main import GPT_MODEL

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


def addtoken_output(num):
    file_path = f"{os.getcwd()}/MATH_TOT_GPT35_output_token.txt"
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
    file_path = f"{os.getcwd()}/MATH_TOT_GPT35_input_token.txt"
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
    file_path = "/Users/natehu/Desktop/Tsinghua Research/ToT复现/TOT/MATH_TOT_4o_token.txt"
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

# def addtoken(num):
#     try:
#         with open("/Users/natehu/Desktop/Tsinghua Research/ToT复现/TOT/tokens.txt", "r") as f:  # 打开文件
#             data = f.read()  # 读取文件
#             nownum = int(data)        
            
#         if num == -1:
#             nownum = 0
#         else:
#             nownum = nownum + num
        
#         with open("tokens.txt","w+") as f:
#             f.write(str(nownum))  # 自带文件关闭功能，不需要再写f.close()
#     except:
#         pass
    
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