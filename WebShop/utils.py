import json
import logging
import math
import os
import random
import sys

import openai
from openai import OpenAI



def askChatGPT(messages, model="gpt-3.5-turbo", temperature = 1, max_tokens=200):
    response = openai.ChatCompletion.create(
            model = model,
            messages = messages,
            temperature = temperature,
            max_tokens = max_tokens,
        )
    addtoken(response.usage.total_tokens)
    answer = response.choices[0].message["content"]
    return answer.strip()


def askLLM(clients, messages, tokens_path, model="gpt-3.5-turbo", temperature = 1, max_tokens=2000,stop=['\n']):
    # 需要包括GPT系列以及LLaMA系列的模型调用,调用接口略有区别
    
    if model in ['gpt-4', 'gpt-4o', 'gpt-3.5-turbo', 'gpt-4o-mini']: # GPT系列模型调用           
        client = clients['gpt']  # gpt系列共用一个client
        response = client.chat.completions.create(
                model = model,
                messages = [
                    {"role": "system", "content": """You are an online webshop agent. You are given an instruction to complete a shopping process. You have to generate the next step in the process. Your action should take into account of the most current obversation (which is the page you are on) and the previous actions taken. 
                     Note: If there is no item that absolutely meets the requirement, choose the one meet at most requirements.

There are two types of actions you can output:
1. search[query]: You can search for a product based on the query. The query is a string that describes the product you are looking for.
2. think[thoughts]: You can think about the current state of the shopping process and decide what to do next.
3. click[button]: You can click on a button on the page to navigate to another page. Where the button are presented in the observation that is bracketed by []. If you think a product is the best choice, you can click on the "Buy Now" button to end the process.

Example of a valid output:
search[noise cancelling cosycost usb microphone]
think[I want to compare the features of the products]
click[Buy Now]
click[< Prev]

Note Don't output Action in front of the action. The action should be in the format of [action][content].
"""},
                    {"role": "user", "content": messages}],
                max_tokens = max_tokens,
                temperature=0.1,
                
                n=1,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=stop
            )
        # print(response.usage

        # addtoken(response.usage.total_tokens)
        update_token_usage(model, response.usage.prompt_tokens, response.usage.completion_tokens, file_path=tokens_path)
        # print(response.usage.prompt_tokens)
        # print(response.usage.completion_tokens)
        answer = response.choices[0].message.content
        
    elif model in ['llama3-8b-8192']:
        client = clients['llama']  # 这里需要改成llama系列的prompts  
        # model = 'Llama3-8B'
        response = client.chat.completions.create(
                model = model,   # 现在可以正常调用llama了
                messages = [
                    {"role": "system", "content": """You are an online webshop agent. You are given an instruction to complete a shopping process. You have to generate the next step in the process. Your action should take into account of the most current obversation (which is the page you are on) and the previous actions taken. 
                    Note: If there is no item that absolutely meets the requirement, choose the one meet at most requirements.

There are two types of actions you can output:
1. search[query]: You can search for a product based on the query. The query is a string that describes the product you are looking for.
2. think[thoughts]: You can think about the current state of the shopping process and decide what to do next.
3. click[button]: You can click on a button on the page to navigate to another page. Where the button are presented in the observation that is bracketed by []. If you think a product is the best choice, you can click on the "Buy Now" button to end the process.

Note Don't output Action in front of the action. The action should be in the format of [action][content].
"""},
                    {"role": "user", "content": messages}],
                temperature = temperature,
                max_tokens = max_tokens,
                stop = stop
            )
        # addtoken(response.usage.total_tokens)
        update_token_usage(model, response.usage.prompt_tokens, response.usage.completion_tokens, file_path=tokens_path)  # 这里就不计算llama的消费了
        answer = response.choices[0].message.content
    else:
        print('MODEL error')
        print(model)
        sys.exit(0)

    return answer.strip()



def askLLM_withprob(clients, messages, tokens_path, model="gpt-3.5-turbo", temperature = 1, max_tokens=200, stop=['\n']):
    # 需要包括GPT系列以及LLaMA系列的模型调用,调用接口略有区别
    probs = {}
    if model in ['gpt-4', 'gpt-4o', 'gpt-3.5-turbo', 'gpt-4o-mini']: # GPT系列模型调用           
        client = clients['gpt']  # gpt系列共用一个client
        response = client.chat.completions.create(
                model = model,
                messages = [
                    {"role": "system", "content": """You are an online webshop agent. You are given an instruction to complete a shopping process. You have to generate the next step in the process. Your action should take into account of the most current obversation (which is the page you are on) and the previous actions taken. If there is no completely fulfiled answer, you can output the most possible answer.

There are two types of actions you can output:
1. search[query]: You can search for a product based on the query. The query is a string that describes the product you are looking for.
2. think[thoughts]: You can think about the current state of the shopping process and decide what to do next.
3. click[button]: You can click on a button on the page to navigate to another page. Where the button are presented in the observation that is bracketed by []. If you think a product is the best choice, you can click on the "Buy Now" button to end the process.

Example of a valid output:
search[noise cancelling cosycost usb microphone]
think[I want to compare the features of the products]
click[Buy Now]
click[< Prev]

Note Don't output Action in front of the action. The action should be in the format of [action][content].
"""},
                    {"role": "user", "content": messages}],
                max_tokens = max_tokens,
                temperature=0.1,
                
                n=1,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=stop,
                logprobs = True
            )
        # print(response.usage
        # add token 需要更加细致.
        # addtoken(response.usage.total_tokens)
        update_token_usage(model, response.usage.prompt_tokens, response.usage.completion_tokens, file_path=tokens_path)
        answer = response.choices[0].message.content
        for item in response.choices[0].logprobs.content:
            # 在这一步就把logprob用e指数返回成prob
            probs[item.token] = math.exp(item.logprob)
        
    elif model in ['llama3-8b-8192']:
        client = clients['llama']  # 这里需要改成llama系列的prompts  # TODO 还没拿到LLaMA的key, 所以先拿gpt-3.5充当.
        # model = 'Llama3-8B'
        response = client.chat.completions.create(
                model = model,   # 现在可以正常调用llama了
                messages = [
                    {"role": "system", "content": """You are an online webshop agent. You are given an instruction to complete a shopping process. You have to generate the next step in the process. Your action should take into account of the most current obversation (which is the page you are on) and the previous actions taken. 

There are two types of actions you can output:
1. search[query]: You can search for a product based on the query. The query is a string that describes the product you are looking for.
2. think[thoughts]: You can think about the current state of the shopping process and decide what to do next.
3. click[button]: You can click on a button on the page to navigate to another page. Where the button are presented in the observation that is bracketed by []. If you think a product is the best choice, you can click on the "Buy Now" button to end the process.

Example of a valid output:
search[noise cancelling cosycost usb microphone]
think[I want to compare the features of the products]
click[Buy Now]
click[< Prev]

Note Don't output Action in front of the action. The action should be in the format of [action][content].
"""},
                    {"role": "user", "content": messages}],
                temperature = temperature,
                max_tokens = max_tokens,
                stop = stop,
                logprobs = True
            )
        # addtoken(response.usage.total_tokens)
        update_token_usage(model, response.usage.prompt_tokens, response.usage.completion_tokens, file_path=tokens_path)
        answer = response.choices[0].message.content
        for item in response.choices[0].logprobs.content:
            probs[item.token] = math.exp(item.logprob)
    else:
        print('MODEL error')
        print(model)
        sys.exit(0)

    return answer.strip(), probs



def update_token_usage(model_name, prompt_tokens, completion_tokens, file_path='token_usage.json'):
    # 读取现有数据
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 如果模型不存在，则初始化模型的数据结构
    if model_name not in data:
        data[model_name] = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
    
    # 更新模型的token数量
    data[model_name]['prompt_tokens'] += prompt_tokens
    data[model_name]['completion_tokens'] += completion_tokens
    data[model_name]['total_tokens'] += (prompt_tokens + completion_tokens)
    
    # 将更新后的数据写回文件
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


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
    if keyid == 0:
        api_key = "your key"
    elif keyid == 1:
        api_key = "your key 1"

    client = OpenAI(api_key=api_key)
    addtoken(-1)
    return client

def printSeq(seq):
    for item in seq:
        print(item)

def judgeNum(num1, num2):
    num1 = num1.replace(',', '')
    num2 = num2.replace(',', '')
    num1 = int(num1)
    num2 = int(num2)
    return 1 if num1 == num2 else 0


def reverseDict(original_dict):
    # 创建一个新的空字典来存储反转后的结果
    reversed_dict = {}

    # 遍历原始字典的键值对
    for key, value in original_dict.items():
        # 如果值已经在反转后的字典中，向其列表中添加键
        if value in reversed_dict:
            reversed_dict[value].append(key)
        else:
            # 否则，创建一个新的列表并将键添加进去
            reversed_dict[value] = [key]
    return reversed_dict

def search_Predecessors(edges, id):
    res = []
    for edge in edges:
        if edge[1] == id:
            res.append(edge[0])
    return res

def setup_logger(tailName=""):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    
    # Create Logs directory if it doesn't exist
    logs_dir = os.path.join(os.getcwd(), "Logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    log_file_path = os.path.join(logs_dir, f"test_{tailName}.log")
    f_handler = logging.FileHandler(log_file_path)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger, f"test_{tailName}.log"

def setup_logger_dot(tailName=""):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    
    # Create Logs directory if it doesn't exist
    logs_dir = os.path.join(os.getcwd(), "Logs/DOT")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    log_file_path = os.path.join(logs_dir, f"test_{tailName}.log")
    f_handler = logging.FileHandler(log_file_path)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger, f"test_{tailName}.log"

def CountCost(token_usage):
    cost_per_1000_tokens = {
        "gpt-4o": {"prompt": 10, "completion": 30},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
        "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
        "gpt-4": {"prompt": 30, "completion": 60}
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


def seconds_to_hms(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return hours, minutes, seconds


def quantile(lst, alpha):
    # 确保 alpha 在 0 和 1 之间
    if not 0 <= alpha <= 1:
        raise ValueError("alpha should be between 0 and 1")

    # 排序列表
    sorted_lst = sorted(lst)

    # 计算分位数的索引
    index = int(alpha * len(sorted_lst))

    # 如果索引等于列表长度，返回最后一个元素
    if index == len(sorted_lst):
        index -= 1

    # 返回分位数的值
    return sorted_lst[index]


def upGradeModel(modelName):
    # gpt-4o-mini, gpt-3.5-turbo, llama3-70b, llama3-8b
    if modelName == 'llama3-8b':
        return 'llama3-70b'
    elif modelName == 'llama3-70b':
        return 'gpt-3.5-turbo'
    elif modelName == 'gpt-3.5-turbo':
        return 'gpt-4o-mini'
    if modelName == 'gpt-4o-mini':
        return 'gpt-4'
    elif modelName == 'gpt-4':
        return 'gpt-4o'
    elif modelName == 'gpt-4o':
        return 'gpt-4o'
        
def allbest_allocation(n):
    selection = {i + 1: 'gpt-4o' for i in range(n)}
    return selection

def random_model_selection(n):
    models = ['gpt-4', 'gpt-4o', 'gpt-3.5-turbo', 'gpt-4o-mini', 'llama3-70b', 'llama3-8b']
    selection = {i + 1: random.choice(models) for i in range(n)}
    return selection

def check_and_create_json_file(file_path):
    # 判断文件是否存在
    if not os.path.exists(file_path):
        # 如果文件不存在，创建一个新文件并写入空的 JSON 结构
        with open(file_path, 'w') as f:
            json.dump({}, f)  # 初始化一个空的 JSON 对象
        # print(f"{file_path} 文件不存在���已创建新文件。")
    else:
        # print(f"{file_path} 文件已存在。")
        pass


def check_and_create_txt_file(file_path):
    # 判断文件是否存在
    if not os.path.exists(file_path):
        # 如果文件不存在，创建一个新文件
        with open(file_path, 'w') as f:
            f.write('')  # 写入空内容创建文件
        # print(f"{file_path} 文件不存在，已创建新文件。")
        return False
    else:
        # print(f"{file_path} 文件已存在。")
        if os.path.getsize(file_path) > 0:
            return True
        else:
            return False
        
        
# 定义模型和数字的映射
model_mapping = {
    'gpt-4o': 5,
    'gpt-4': 4,
    'gpt-4o-mini': 3,
    'gpt-3.5-turbo': 2,
    'llama3-70b': 1,
    'llama3-8b': 0
}

def downgrading_vanilla(original_dict):
    # 第一步：转换value为映射后的数字
    reverse_mapping = {v: k for k, v in model_mapping.items()}  # 创建反向映射
    converted_dict = {k: model_mapping[v] for k, v in original_dict.items()}

    # 筛选出值大于 0 的键
    keys_above_min = [k for k, v in converted_dict.items() if v > 0]
    if len(keys_above_min)==0:
        return False
        
    # 如果存在大于 0 的值
    if keys_above_min:
        # 随机选择 1-3 个键
        num_keys_to_decrement = random.choice([1, 2])
        keys_to_decrement = random.sample(keys_above_min, min(num_keys_to_decrement, len(keys_above_min)))

        # 将选中的键对应的值减 1
        for key in keys_to_decrement:
            converted_dict[key] -= 1

    # 第三步：将数字转换回模型名称
    final_dict = {k: reverse_mapping[v] for k, v in converted_dict.items()}

    return final_dict


def downgrading_pro(original_dict):
    # 第一步：转换value为映射后的数字
    reverse_mapping = {v: k for k, v in model_mapping.items()}  # 创建反向映射
    converted_dict = {k: model_mapping[v] for k, v in original_dict.items()}

    # 筛选出值大于 0 的键
    keys_above_min = [k for k, v in converted_dict.items() if v > 1]
    if len(keys_above_min)==0:
        return False
        
    # 如果存在大于 0 的值
    if keys_above_min:
        # 随机选择 1-3 个键
        num_keys_to_decrement = random.choice([2, 3, 4])
        keys_to_decrement = random.sample(keys_above_min, min(num_keys_to_decrement, len(keys_above_min)))

        # 将选中的键对应的值减 2
        for key in keys_to_decrement:
            converted_dict[key] -= 2

    # 第三步：将数字转换回模型名称
    final_dict = {k: reverse_mapping[v] for k, v in converted_dict.items()}

    return final_dict


def downgrading_promax(original_dict):
    # 第一步：转换value为映射后的数字
    reverse_mapping = {v: k for k, v in model_mapping.items()}  # 创建反向映射
    converted_dict = {k: model_mapping[v] for k, v in original_dict.items()}

    # 筛选出值大于 0 的键
    keys_above_min = [k for k, v in converted_dict.items() if v > 0]
    if len(keys_above_min)==0:
        return False
        
    # 如果存在大于 0 的值
    if keys_above_min:
        # 随机选择 1-3 个键
        num_keys_to_decrement = random.choice([2, 3, 4])
        keys_to_decrement = random.sample(keys_above_min, min(num_keys_to_decrement, len(keys_above_min)))

        # 将选中的键对应的值减 1
        for key in keys_to_decrement:
            converted_dict[key] -= 1

    # 第三步：将数字转换回模型名称
    final_dict = {k: reverse_mapping[v] for k, v in converted_dict.items()}

    return final_dict


def upgrading(original_dict):
    # 第一步：转换value为映射后的数字
    reverse_mapping = {v: k for k, v in model_mapping.items()}  # 创建反向映射
    converted_dict = {k: model_mapping[v] for k, v in original_dict.items()}

    # 筛选出值小于 5 的键
    keys_below_max = [k for k, v in converted_dict.items() if v < 5]

    # 如果存在小于 5 的值
    if keys_below_max:
        # 随机选择 1-2 个键
        num_keys_to_increment = random.choice([1, 2])
        keys_to_increment = random.sample(keys_below_max, min(num_keys_to_increment, len(keys_below_max)))

        # 将选中的键对应的值加 1
        for key in keys_to_increment:
            converted_dict[key] += 1

    # 第三步：将数字转换回模型名称
    final_dict = {k: reverse_mapping[v] for k, v in converted_dict.items()}

    return final_dict


# 定义将数据写入JSON文件的函数
def write_json(file_path, data):
    try:
        # 打开文件并写入数据，确保格式化输出
        # 主义是覆盖写入,每次写入会刷掉之前的格式
        with open(file_path, 'w', encoding='utf-8') as f:  
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"数据成功写入 {file_path}")
    except Exception as e:
        print(f"发生错误：{e}")
        
        
def write_json_listoneline(file_path, data):
    try:
        # 自定义递归函数，用于处理 list 和其他类型的数据
        def custom_json_encoder(obj, indent=0):
            # 定义缩进
            indent_str = ' ' * indent
            
            if isinstance(obj, dict):
                # 处理 dict 类型
                json_str = '{\n'
                for i, (key, value) in enumerate(obj.items()):
                    if i > 0:
                        json_str += ',\n'
                    json_str += f'{indent_str}  "{key}": {custom_json_encoder(value, indent + 2)}'
                json_str += f'\n{indent_str}}}'
                return json_str

            elif isinstance(obj, list):
                # 处理 list 类型，不换行
                return json.dumps(obj, separators=(',', ':'))

            else:
                # 处理其他类型
                return json.dumps(obj, ensure_ascii=False)

        # 打开文件并写入数据
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(custom_json_encoder(data))
        
        print(f"数据成功写入 {file_path}")
    except Exception as e:
        print(f"发生错误：{e}")


def extract_numbers_from_filenames(folder_path):
    numbers = []
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            # 提取文件名中的数字部分并转换为整数
            number = int(file_name.split('.')[0])
            numbers.append(number)
    return numbers


def sort_with_indices(input_list):
    # 使用enumerate为每个��素加上原始索引，然后根据值进行排序
    sorted_indices = sorted(range(len(input_list)), key=lambda k: input_list[k], reverse=True)
    sorted_list = [input_list[i] for i in sorted_indices]  # 根据排序后的索引生成排序后的列表
    return sorted_list, sorted_indices


def find_first_valid_key(lst, dictx):
    for key in lst:
        if dictx.get(key) == 'gpt-4o':  # 检查key在dictx中的值
            return key
    return None  # 如果没有找到符合条件的返回None


def find_first_valid_key2(lst, dictx):
    for key in reversed(lst):
        if dictx.get(key) == 'llama3-8b':  # 检查key在dictx中的值
            return key
    return None  # 如果没有找到符合条件的返回None

if __name__ == '__main__':
    # print(judgeNum(1,1))
    # print(search_Predecessors( [(1, 4), (1, 3), (1, 2),(2,5) ,(3, 5), (5, 6), (6, 7)], 5))
    lst = [1,2,3,4,5,6,7,8,9,10]
    print(quantile(lst, 0.2))  # 做过极限测试,0和1都是可行的
    
    # # 原始字典
    # original_dict = {1: 'llama3-8b', 2: 'gpt-4o-mini', 3: 'llama3-8b', 4: 'gpt-4o-mini', 5: 'gpt-4', 6: 'gpt-4o-mini', 7: 'gpt-4'}
    # # 调用函数处理字典
    # result = upgrading(original_dict)
    # # 输出结果
    # print(result)



