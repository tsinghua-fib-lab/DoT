import sys
from collections import defaultdict, deque

sys.path.append('C:\\Users\\Pluto\\Desktop\\TaDe')
import re
from typing import Any, List

import ipydagred3
import networkx as nx

from utils import *

GPT_MODEL = "gpt-4-turbo"

def extract_in_out(text):
    # 使用正则表达式提取IN和OUT后的内容
    match = re.search(r'IN:\s*(.*?)\s*OUT:\s*(.*)', text)
    question = match.group(1)
    actions = match.group(2)
    return question, actions
    

def decompose_sql(clients, passage, question, config):    


    prompt_for_decompose = f"""
Here is a math word problem. I will first provide a passage of the problem to set the context. Then, I will ask a specific question that requires you to use the information from the problem description, along with calculation and reasoning, to solve it. 
However, before solving the specific question, I need you to break it down into several sub-questions.

Passage: {passage}

Question: {question}

Please decompose the question into smaller sub-questions.
Answer Format: (Please write each broken-down question step on a separate line, starting with a number.)
To solve the question "xxx", we need to know:
"1. question step 1",
"2. question step 2",
"3. question step 3".
...
"""
    Q = {
        "role": "user",
        "content": prompt_for_decompose
    }
    # Query = Example+[Q]
    Query = [Q]
    # result = askChatGPT(Query, model=GPT_MODEL, temperature=1)
    result = askLLM(clients, Query, tokens_path=config['tokens_path'], model=config['decompose_MODEL'], temperature=1)
    return result


def convert_steps_to_format(decom_commands):
    # 截取“we need to know:”后的内容
    start_index = decom_commands.find("we need to know:") + len("we need to know:")
    subtasks_text = decom_commands[start_index:].strip()

    # 将每个子任务单独列出
    subtasks = subtasks_text.split('\n')
    subtasks = [task.strip().split('. ', 1)[-1] for task in subtasks]
    
    steps_dict = {index + 1: value for index, value in enumerate(subtasks)}
    
    return subtasks, steps_dict


def AllocateModel(clients, passage, question, steps, config):
    
    # 在这里就进行分配,确定用哪个大语言模型来解决问题
    # 一次性对所有的子任务分配MODEL
    # Below are some examples of model assignments based on the difficulty of the problems themselves.
    allo_Q = [{'role':'user', 'content':f"""I am now going to solve a math problem. I have broken down the initial problem into several smaller ones and hope to solve these small problems step by step using large language models(LLMs).
Now I need to assign a LLM to each small problem, which can be one of the following: [gpt-4-turbo, gpt-4, gpt-4o-mini, gpt-3.5-turbo, llama3-70b, llama3-8b].
Their capabilities are ranked from strongest to weakest as follows: [gpt-4-turbo, gpt-4, gpt-4o-mini, gpt-3.5-turbo, llama3-70b, llama3-8b]
Their costs are ranked from highest to lowest as follows: [gpt-4, gpt-4-turbo, gpt-3.5-turbo, gpt-4o-mini, llama3-70b, llama3-8b]

Now the background of the mathematical problem is: \n{passage}\n
The math problem is: \n{question}\n
The list of subtasks is as follows: \n{steps}\n

Please help me evaluate the difficulty of each subtask (using an integer score from 1 to 5) and assign a dedicated MODEL to each.
Output format: (No additional explanations or comments are needed. Please strictly follow the format below in your response. Place the content, difficulty, and assigned model of each sub-problem in a tuple, and put all the tuples in a list.) 
[
    ("subtask1", "difficulty1", "MODEL1"),
    ("subtask2", "difficulty2", "MODEL2"),
    ("subtask3", "difficulty3", "MODEL3"),
    ...
]
"""}]
    # allo_model = askChatGPT(allo_Q, model=config['allocation_MODEL'], temperature=1)  # 询问大语言模型这个子任务应该使用什么样的模型来完成
    allo_model = askLLM(clients, allo_Q, tokens_path=config['tokens_path'], model=config['allocation_MODEL'], temperature=1)
    # print('allocate model to each:\n')
    # print(allo_model)
    # print('\n')
    allo_model = eval(allo_model)
    allo_model = {index+1: tup[2] for index, tup in enumerate(allo_model)}  # 这里的index需要和steps_dict保持一致,从1开始.
    return allo_model



def construct_dependencies_without_traversal(clients, question, steps, config):
    # 这里的steps只是一个list
    
    connect_system = {"role": "system", "content": """Now we have a math problem, which we have broken down into many sub-problems. I want you to understand the connection between these sub-problems."""}
    connect_user = {"role": "user", "content":f"The init math problem is {question}. And the sub-problems are {steps}. Please provide your understanding of the relationships between these sub-problems. Your response must be concise."}
    connect_Q = [connect_system, connect_user]
    # connect_commands = askChatGPT(connect_Q, model=GPT_MODEL, temperature = 1)
    connect_commands = askLLM(clients, connect_Q, tokens_path=config['tokens_path'], model=config['dependencies_1_MODEL'], temperature=1)
    # print('LLM对于subcommands之间关系的语言理解:\n\n', connect_commands)
    connect_Q.append({
        "role": "assistant",
        "content": connect_commands,
    })

    prompt = f"""Now we need to create standardized connections for the relationships between these sub-problems.
    Now Given the following subtasks for question: {question}, determine the dependencies between them:\n"""

    for count, step in enumerate(steps, start=1):
        prompt += f"Step {count}: {step}\n"
        
    prompt += """\nPlease list the dependencies in the format 'Subproblem A [xxx] -> Subproblem B [xxx]' indicating that Sub-problem A must be completed before Sub-problem B can start.
Please identify any potential conditional dependencies from a logical perspective.

Answer format: (Please strictly follow the format. Each dependency should be separated by a new line. No explanation is required.)
Step id_i [ sub-problem i ] -> Step id_j [ sub-problem j ]
Step id_m [ sub-problem m ] -> Step id_n [ sub-problem n ]
...
"""
    prompt_user = {
        "role": "user",
        "content": f"""{prompt}"""
    }
    Query = [prompt_user]
    # output = askChatGPT(Query, model=GPT_MODEL, temperature=1)
    output = askLLM(clients, Query, tokens_path=config['tokens_path'], model=config['dependencies_2_MODEL'], temperature=1)
    return output

def create_dag_from_string(dependencies_string):
    G = nx.DiGraph()
    
    # Split the string by newlines to get individual dependencies
    dependencies = dependencies_string.strip().split('\n')
    
    for dependency in dependencies:
        # Split each dependency by the arrow
        steps = dependency.split(' -> ')
        if len(steps) == 2:
            step_0 = steps[0].strip()
            step_1 = steps[1].strip()
            # Add edges based on dependencies
            G.add_edge(step_0, step_1)
    
    # Print the edges
    # print(list(G.edges()))
    # Perform transitive reduction to simplify the graph
    transitive_reduction = nx.transitive_reduction(G)
    return transitive_reduction

def create_graph(reduced_dependencies, problem):
    graph = ipydagred3.Graph()

    #for state in steps_list:
    #    graph.setNode(state, tooltip='tooltip1 of ' + state)
        
    for dependency in reduced_dependencies:
        graph.setEdge(*dependency)

    graph.setNode(problem, tooltip = "Problem Statement")
        
    widge_try = ipydagred3.DagreD3Widget(graph=graph)
    
    return widge_try

def calculate_node_depths(edges):
    # 初始化节点的入度和邻接表
    in_degree = defaultdict(int)
    adjacency_list = defaultdict(list)
    
    nodes = set()
    
    # 填充邻接表和计算入度
    for u, v in edges:
        adjacency_list[u].append(v)
        in_degree[v] += 1
        nodes.add(u)
        nodes.add(v)
    
    # 找到所有入度为0的节点
    queue = deque([node for node in nodes if in_degree[node] == 0])
    depth = {node: 0 for node in queue}
    
    # 广度优先搜索
    while queue:
        node = queue.popleft()
        current_depth = depth[node]
        
        for neighbor in adjacency_list[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                depth[neighbor] = current_depth + 1
    
    return depth

# 定义类型字典
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
    
    
def sentenceRes2Actions(sentence):
    
    rewrite_system = {"role": "system", "content": f"""Now I have a pseudo action sequence expression with parentheses and multiplication. I need you to help me convert this into a sequence of actions without an operator sign.
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
    actions = askChatGPT(Q_now, model=GPT_MODEL, temperature = 1)

    # 删除数字
    text_no_numbers_and_brackets = re.sub(r'[\d\(\)]', '', actions)
    # 替换 TURN 后面的空格为下划线
    text_formatted = re.sub(r'TURN\s+', 'TURN_', text_no_numbers_and_brackets)
    # 去掉多余的空格
    text_formatted = re.sub(r'\s+', ' ', text_formatted).strip()
    return text_formatted

def solve_Sub_Question(clients, answer_MODEL, passage, question, answerDict, int_edges, number, subtask, tokens_path, alpha):
    # 求解子任务
    sys_q = f"""Here is a math word problem. I will first provide a passage of the problem to set the context. Then, I will ask a specific question that requires you to use the information from the problem description, along with calculation and reasoning, to solve it. 
Passage:\n{passage}

Question:\n{question}

I have broken this math question down into several smaller questions. I will assign you sub-questions one by one, and provide the results of the previous sub-questions as a reference for your reasoning.
Please solve the question according to mathematical logic.
"""  # 系统任务信息
    
    if len(answerDict)>0:
        answersSoFar = f"""\nSo far, the answers to the resolved sub-questions are as follows: The format is Sub-question-Id: xxx; Sub-question: xxx; Answer: xxx."""
        for key, value in answerDict.items():
            answersSoFar += f"""\nSub-question-Id: {key}; Sub-question: {answerDict[key]['subtask']}; Answer: {answerDict[key]['answer']}."""
        
        predecessors = search_Predecessors(int_edges, number)
        intersection = set(answerDict.keys()).intersection(set(predecessors))
        count = len(intersection)
        if count>0:
            answersSoFar += f"""\nAmong them, sub-questions {predecessors} are directly related to this sub-question, so please pay special attention to them."""
    
    
    subask = f"""\nThe sub-question to solve now is xxx: {subtask}
Based on the information above, please provide a concise and clear answer"""

    if len(answerDict)>0:
        query = answersSoFar+subask
    else:
        query = subask

    Q = [{'role':'system', 'content':sys_q},
        {'role':'user', 'content':query},]
        
    # result = askChatGPT(Q, model='gpt-3.5-turbo', temperature=1, max_tokens=300)  # 主动限制回答长度
    result, probs = askLLM_withprob(clients, Q, tokens_path=tokens_path, model=answer_MODEL, temperature=1, max_tokens=500)
    prob_values = list(probs.values())
    frac = quantile(prob_values, alpha)
    return result, frac, Q

if __name__ == '__main__':
    
    import pickle
    file = open('finalResult.pkl','rb') 
    finalResult = pickle.load(file)

    print(finalResult)
    print(finalResult == "\"[5506558, 2]\"")
    
    finalResult = remove_quotes(finalResult)
    
    # answer = convert_result(finalResult, 'List[int]')
    # print(type(answer))
    finalResult = convert_to_type('List[int]', finalResult)
    print(type(finalResult))
    
    
    