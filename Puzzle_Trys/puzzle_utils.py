import sys
from collections import defaultdict, deque

import ipydagred3
import networkx as nx

sys.path.append('C:\\Users\\Pluto\\Desktop\\TaDe')
from typing import Any, List

from utils import *

GPT_MODEL = "gpt-4-turbo"

def decompose_sql(clients, question, config):    
    
#     question_example = f"""
# You will be provided with a Programming Puzzle. Your task is to find an input that will make the program return True.
# Here is the puzzle: 
# def sat(start: int, k=1, lower=93, seq=[-61, -46, 89, 93, -13, 14, -95, -74, -92, -38, -93, 64, -78, 3, 92, -10, -4, 43, 72, 12, 3, -3, -15, -96, 72, -71, -30, 53, 17, -87, 49, 17, -69, 78, 6, -77, -99, 91, 13, 9, 81, -55, 75, 48, -65, 18, -83, 10, -12, 88, 60, -72, -7, -49, -56, -76, 82, 18, 77, 52, -92, -88, 39, 13, -16, 82, 4, 44, -19, 54, 6, 55, 77, -38, -30, -55, -16]):
#     return 0 <= start <= len(seq) - k and sum(seq[start:start + k]) >= lower
    
# You have to decompose the puzzle into multiple steps. Carefully consider the granularity of each subtask to ensure that each one is executable.

# Answer format: Please follow the format below strictly when answering. No explanation is required.
# STEP1 [ step task 1 ]
# STEP2 [ step task 2 ]
# ...
# """
#     answer_example = f"""
# STEP1 [ Understand the puzzle's primary objective: Find an integer 'start' such that when added to the subsequent 'k' elements in 'seq', the sum is greater than or equal to 'lower'. ]
# STEP2 [ Note the inputs: 'k', 'lower', and 'seq' are given and fixed. ]
# STEP3 [ Recognize the constraints: The variable 'start' must be within the range 0 to len(seq) - k. ]
# STEP4 [ Comprehend that sum(seq[start:start + k]) needs to be greater than or equal to 'lower'. ]
# STEP5 [ Deduce a strategy: Loop through the sequence from 0 to len(seq) - k to find a valid 'start'. ]
# STEP6 [ For each 'start' value in the loop, calculate sum(seq[start:start + k]). ]
# STEP7 [ Compare the calculated sum with 'lower'. ]
# STEP8 [ If the sum is greater than or equal to 'lower', return that 'start' value. ]
# """

#     Example = [
#         {"role": "user", "content": question_example},
#         {"role": "assistant", "content": answer_example}
#     ]


    prompt_for_decompose = f"""You will be provided with a Programming Puzzle. The ultimate task is to find an input that will make the program return True.
To better accomplish this task, now you need to break the puzzle into multiple steps, preferably between 3 and 8 steps.
4 examples are as follows:

Program 1:
def sat(li: List[int], k=6):
    def prod(nums):
        ans = 1
        for i in nums:
            ans *= i
            return ans
    return min(li) > 1 and len(li) == k and all((1 + prod(li[:i] + li[i + 1:])) % li[i] == 0 for i in range(k))    
Result 1 of decomposed steps:
STEP 1 [ Understand the conditions required by the function. ]
STEP 2 [ Choose the length of the list based on k. ]
STEP 3 [ Generate potential elements for the list. ]
STEP 4 [ Calculate the product of all other elements for each element in the list when i = 0 and add 1 to the product. ]
STEP 5 [ Calculate the product of all other elements for each element in the list when i = 1 and add 1 to the product. ]
STEP 6 [ Calculate the product of all other elements for each element in the list when i = 2 and add 1 to the product. ]
STEP 7 [ Calculate the product of all other elements for each element in the list when i = 3 and add 1 to the product. ]
STEP 8 [ Calculate the product of all other elements for each element in the list when i = 4 and add 1 to the product. ]
STEP 9 [ Calculate the product of all other elements for each element in the list when i = 5 and add 1 to the product. ]
STEP 10 [ Verify the divisibility condition for each element. ]
STEP 11 [ Adjust the elements and repeat until a valid list is found. ]
STEP 12 [ Confirm that the list meets all conditions. ]


Program 2:
def sat(indices: List[int], s=\"aeEm%%uIV0imR&xUvQvZf#1z4\"):
    i, j = indices
    return s[i] == s[j] and 0 <= i < j < i + 3
Result 2 of decomposed steps:
STEP 1 [ Understand there are two conditions need to fulfill for the input indices that i and j in the indices should meet first s[i] == s[j] and 0 <= i < j < i + 3 ]
STEP 2 [ Iterate through the string sin a group of 3 characters, s[n] s[n+1] s[n+2] ]
STEP 3 [ Compare the three characters to see if any of two characters are the same. ]
STEP 4 [ If identical strings are found, Count the index of both % in the string s ]
STEP 5 [ If no identical characters, move to the consecutive three characters. ]
STEP 6 [ Write the index of two identical characters and yield the final answer of list indices. ]


Program 3:
def sat(path: List[int], weights=[{{1: 20, 2: 1}}, {{2: 2, 3: 5}}, {{1: 10}}], bound=11):
    return path[0] == 0 and path[-1] == 1 and sum(weights[a][b] for a, b in zip(path, path[1:])) <= bound
Result 3 of decomposed steps:
STEP 1 [ The input path should be a list of integers with three constraints need to fulfill. ]
STEP 2 [ Create a list that fulfill the first contraint to have 0 at index 0. ]
STEP 3 [ Create a list that fulfill the second contraint to have 1 at last. ]
STEP 4 [ Given that the sum of weights[a][b] for a, b in zip(path,path[1:])) <= bound, we need to find values in the list weightsthat is less than 11. ]
STEP 5 [ First checking if combining step 2 and step 3 to be path could be the correct input by calculating sum(weights[a][b] for a, b in zip(path, path[1:])) <= bound ]
STEP 6 [ If the previous step is not correct, then think about what could be the integer filling between 0 and 1. ]
STEP 7 [ Eliminate the incorrect candidates. ] 
STEP 8 [ Fill in the number to the list of integer. ]
STEP 9 [ Verify the if the new list will make the function return True. ]
STEP 10 [ Output the answer ]

Program 4:
"name": "LastLetters:3",
def sat(y: List[bool], x=['ryxadec', 'pyfixotibujadyxe', 
                          'mopubywewexi witethig 7', ' !', 
                          'jethi sed c', 'lotextusavufubynyb',
                          'wuxesafetatextysima pebutextiwafufok',
                          'tuchonip', ' S', 
                          'xyvovikofutex pylekazuquekedajota E', 
                          'wik xofoxujegerigubo ?', 
                          'gipimakude 1', ' O', ' ^', 
                          'lakiquuvuhenugu vajyquy P', 
                          ' 6', 'fezore', 'vabithin textusichytilejocoke',
                          ' B', 'lasuthasebuvy que &',
                          'mymanuzuzudyc thazufys y', '', ' ?',
                          'gecohywelawu', 'wath']):
    assert len(x) == len(y)
    for s, b in zip(x, y):
        if len(s.split(" ")[-1]) == 1:
            assert b == s[-1].isalpha()
        else:
            assert not b
    return True
Result 4 of decomposed steps:
STEP 1 [ Determine the length of the list x to ensure yhave the same length. ]
STEP 2 [ Loop through the list x to check the last word of string. ]
STEP 3 [ Check if the last segment of the string in x (seperated by space) have length 1 ]
STEP 4 [ If Step 3 meet, check if that character is alphabetical characters. ]
STEP 5 [ If step 4 is true, then the boolean value in list y with corresponding index should also be True. If not, False. ]
STEP 6 [ If Step 3 do not meet, the boolean value in list y with corresponding index should be False. ]
STEP 7 [ The final result should a list of boolean values. ]


Now here is the puzzle for you to decompose: {question}
Requirements:
1. The steps broken down should preferably be between 3 to 8 steps.
2. Each step needs to be executable, have a clear meaning, or produce a meaningful result.

Answer format: Please follow the format below strictly when answering. No explanation is required.
STEP 1 [ step task 1 ]
STEP 2 [ step task 2 ]
...
"""
    Q = {
        "role": "user",
        "content": prompt_for_decompose
    }
    Query = [Q]
    # Query = [Q]
    # result = askChatGPT(Query, model=GPT_MODEL, temperature=1)
    result = askLLM(clients, Query, tokens_path=config['tokens_path'], model=config['decompose_MODEL'], temperature=1)
    return result


def convert_steps_to_format(raw_steps):
    lines = raw_steps.strip().split('\n')
    steps_dict = {}
    steps = []
    for line in lines:
        if line.strip():  # 只处理非空行
            step_number = int(line.split(' ')[1])  # 提取数值部分并转换为整数
            # step_id = line.split(' ')[0]
            step_content = line[line.index('[') + 1 : line.rindex(']')]
            steps_dict[step_number] = step_content
            steps.append({"stepId": step_number, "step": step_content})
            
            
    # return steps_dict
    return steps, steps_dict


def AllocateModel(clients, question, steps, config):
    allo_Q = [{'role':'user', 'content':f"""Here is a Programming Puzzle and I want to find an input that will make the program return True.
I have already broken down the solution process into several smaller steps and hope you can help assign an LLM to solve each of these steps.
LLMs can be chosen from: [gpt-4-turbo, gpt-4, gpt-4o-mini, gpt-3.5-turbo, llama3-70b, llama3-8b].
Their capabilities are ranked from strongest to weakest as follows: [gpt-4-turbo, gpt-4, gpt-4o-mini, gpt-3.5-turbo, llama3-70b, llama3-8b]
Their costs are ranked from highest to lowest as follows: [gpt-4, gpt-4-turbo, gpt-3.5-turbo, gpt-4o-mini, llama3-70b, llama3-8b]

Now, my original Programming Puzzle is: \n{question}\n
The list of substeps is as follows: \n{steps}\n
Please help me evaluate the difficulty of each substep (using an integer score from 1 to 5) and assign a dedicated MODEL to each.
Output format: (No additional explanations or comments are needed. Please strictly follow the format below in your response. Place the content, difficulty, and assigned model of each step in a tuple, and put all the tuples in a list.) 
[
    ("step1 content", "difficulty1", "MODEL1"),
    ("step2 content", "difficulty2", "MODEL2"),
    ("step3 content", "difficulty3", "MODEL3"),
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
    prompt = f"Now Given the following subtasks for question: {question}, determine the dependencies between them:\n"
    for step in steps:
        prompt += f"Step {step['stepId']}: {step['step']}\n"
    prompt += """\nPlease list the dependencies in the format 'Subtask A [xxx] -> Subtask B [xxx]' indicating that Subtask A must be completed before Subtask B can start.
Please identify any potential conditional dependencies from a logical perspective.

Answer format: (Please strictly follow the format. Each dependency should be separated by a new line. No explanation is required.)
Step id_i [  subtask i ] -> Step id_j [ subtask j ]
Step id_m [  subtask m ] -> Step id_n [ subtask n ]
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


def solve_Sub_Question(clients, answer_MODEL, question, answer_type, answerDict, int_edges, number, subtask, tokens_path, alpha):
    # question 问题字符串
    # 交待解决任务
    sys_q = f"""You will be provided with a Programming Puzzle. Your task is to find an input that will make the program return True.
Here is the puzzle:\n{question}

The data type of your final answer should be {answer_type}.
I have broken this puzzle down into many easier subtasks. I will assign you sub-tasks one by one, and provide the results of the previous sub-tasks as a reference for your reasoning.
Please follow the logical sequence of our subtasks to find the correct input."""
    
    if len(answerDict)>0:
        answersSoFar = f"""\nSo far, the answers to the resolved sub-tasks are as follows: The format is SubtaskId: xxx; Subtask: xxx; Answer: xxx."""
        for key, value in answerDict.items():
            answersSoFar += f"""\nSubtaskId: {key}; Subtask: {answerDict[key]['subtask']}; Answer: {answerDict[key]['answer']}."""
        
        predecessors = search_Predecessors(int_edges, number)
        intersection = set(answerDict.keys()).intersection(set(predecessors))
        count = len(intersection)
        if count>0:
            answersSoFar += f"""\nAmong them, sub-tasks {predecessors} are directly related to this sub-task, so please pay special attention to them."""
    
    
    subask = f"""\nNow the subtask is: {subtask}
Based on the information above, please provide a concise and clear answer to this sub-task in one or two sentences.."""

    if len(answerDict)>0:
        query = answersSoFar+subask
    else:
        query = subask

    Q = [{'role':'system', 'content':sys_q},
        {'role':'user', 'content':query},]
        
    # print(subtaskid)
    # print(subtask)
    # print('**********Question**********')
    # print(Q)
    # result = askChatGPT(Q, model='gpt-3.5-turbo', temperature=1)
    result, probs = askLLM_withprob(clients, Q, tokens_path=tokens_path, model=answer_MODEL, temperature=1, max_tokens=300)
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
    
    
    