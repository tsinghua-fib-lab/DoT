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


def askLLM(clients, messages, tokens_path, model="gpt-3.5-turbo", temperature = 1, max_tokens=2000):
    # 需要包括GPT系列以及LLaMA系列的模型调用,调用接口略有区别
    
    if model in ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o']: # GPT系列模型调用           
        client = clients['gpt']  # gpt系列共用一个client
        response = client.chat.completions.create(
                model = 'gpt-4o',
                messages = messages,
                temperature = temperature,
                max_tokens = max_tokens,
            )
        # print(response.usage
        # add token 需要更加细致.
        # addtoken(response.usage.total_tokens)
        model ='gpt-4o'
        update_token_usage(model, response.usage.prompt_tokens, response.usage.completion_tokens, file_path=tokens_path)
        # print(response.usage.prompt_tokens)
        # print(response.usage.completion_tokens)
        answer = response.choices[0].message.content
        
    elif model in ['llama3-70b', 'llama3-8b']:
        client = clients['llama']  # 这里需要改成llama系列的prompts  # TODO 还没拿到LLaMA的key, 所以先拿gpt-3.5充当.
        model = model+'-8192'
        # model = 'Llama3-8B'
        response = client.chat.completions.create(
                model = model,   # 现在可以正常调用llama了
                messages = messages,
                temperature = temperature,
                max_tokens = max_tokens,
            )
        # addtoken(response.usage.total_tokens)
        # update_token_usage("gpt-3.5-turbo", response.usage.prompt_tokens, response.usage.completion_tokens, file_path=tokens_path)  # 这里就不计算llama的消费了
        answer = response.choices[0].message.content
    else:
        print('MODEL error')
        print(model)
        sys.exit(0)

    return answer.strip()



def askLLM_withprob(clients, messages, tokens_path, model="gpt-3.5-turbo", temperature = 1, max_tokens=200):
    # 需要包括GPT系列以及LLaMA系列的模型调用,调用接口略有区别
    probs = {}
    if model in ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-4o-mini']: # GPT系列模型调用           
        client = clients['gpt']  # gpt系列共用一个client
        response = client.chat.completions.create(
                model = model,
                messages = 'gpt-4o',
                temperature = temperature,
                max_tokens = max_tokens,
                logprobs = True,
            )
        # print(response.usage
        # add token 需要更加细致.
        # addtoken(response.usage.total_tokens)
        model ='gpt-4o'
        update_token_usage(model, response.usage.prompt_tokens, response.usage.completion_tokens, file_path=tokens_path)
        answer = response.choices[0].message.content
        for item in response.choices[0].logprobs.content:
            # 在这一步就把logprob用e指数返回成prob
            probs[item.token] = math.exp(item.logprob)
        
    elif model in ['llama3-70b', 'llama3-8b']:
        client = clients['gpt']  # 这里需要改成llama系列的prompts  # TODO 还没拿到LLaMA的key, 所以先拿gpt-3.5充当.
        response = client.chat.completions.create(
                model = "gpt-3.5-turbo",  # TODO 还没拿到LLaMA的key, 所以先拿gpt-3.5充当.
                messages = messages,
                temperature = temperature,
                max_tokens = max_tokens,
                logprobs = True,
            )
        # addtoken(response.usage.total_tokens)
        update_token_usage("gpt-3.5-turbo", response.usage.prompt_tokens, response.usage.completion_tokens, file_path=tokens_path)
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
        with open("tokens.txt", "r") as f:
            data = f.read()
            nownum = int(data)        
            
        if num == -1:
            nownum = 0
        else:
            nownum = nownum + num
        
        with open("tokens.txt","w+") as f:
            f.write(str(nownum))
    except:
        pass

    
def setOpenAi(keyid = 0):
    # set your openai key here.
    if keyid == 0:
        api_key = ""
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
    reversed_dict = {}

    # 遍历原始字典的键值对
    for key, value in original_dict.items():
        if value in reversed_dict:
            reversed_dict[value].append(key)
        else:
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
    f_handler = logging.FileHandler("Logs/test_"+tailName+".log")

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger, "test_"+tailName+".log"

def CountCost(token_usage):
    cost_per_1000_tokens = {
        "gpt-4-turbo": {"prompt": 10, "completion": 30},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
        "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
        "gpt-4": {"prompt": 30, "completion": 60},
        "gpt-4o": {"prompt": 2.5, "completion": 10}
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
        return 'gpt-4-turbo'
    elif modelName == 'gpt-4-turbo':
        return 'gpt-4-turbo'
        
def allbest_allocation(n):
    selection = {i + 1: 'gpt-4-turbo' for i in range(n)}
    return selection

def random_model_selection(n):
    models = ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-4o-mini', 'llama3-70b', 'llama3-8b']
    selection = {i + 1: random.choice(models) for i in range(n)}
    return selection

def check_and_create_json_file(file_path):
    # 判断文件是否存在
    if not os.path.exists(file_path):
        # 如果文件不存在，创建一个新文件并写入空的 JSON 结构
        with open(file_path, 'w') as f:
            json.dump({}, f)  # 初始化一个空的 JSON 对象
        # print(f"{file_path} 文件不存在，已创建新文件。")
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
    'gpt-4-turbo': 5,
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
    # 使用enumerate为每个元素加上原始索引，然后根据值进行排序
    sorted_indices = sorted(range(len(input_list)), key=lambda k: input_list[k], reverse=True)
    sorted_list = [input_list[i] for i in sorted_indices]  # 根据排序后的索引生成排序后的列表
    return sorted_list, sorted_indices


def find_first_valid_key(lst, dictx):
    for key in lst:
        if dictx.get(key) == 'gpt-4-turbo':  # 检查key在dictx中的值
            return key
    return None  # 如果没有找到符合条件的返回None


def find_first_valid_key2(lst, dictx):
    for key in reversed(lst):
        if dictx.get(key) == 'llama3-8b':  # 检查key在dictx中的值
            return key
    return None  # 如果没有找到符合条件的返回None

if __name__ == '__main__':
    lst = [1,2,3,4,5,6,7,8,9,10]
    print(quantile(lst, 0.2)) 

