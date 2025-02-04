import sys
from collections import defaultdict, deque

sys.path.append('C:\\Users\\Pluto\\Desktop\\TaDe')
import re
from typing import Any, List

import ipydagred3
import networkx as nx

from utils import *


def extract_in_out(text):
    # 使用正则表达式提取IN和OUT后的内容
    match = re.search(r'IN:\s*(.*?)\s*OUT:\s*(.*)', text)
    question = match.group(1)
    actions = match.group(2)
    return question, actions
    

'''
还没有转化成例子的4个例子如下:
Question 5:
How many subsets of {{1, 2,..., 10}} have no two successive numbers?
Answer 5:
The process of solving the problem can be divided into the following steps:
1. For each subset of the set {{1, ..., n}}, we map it to a length-n string of 0s and 1s, where a digit 1 at i-th place means that i is in the subset.
2. Thus, the constraint that the subset has no successive numbers means that the string has no two consecutive 1s.
3. Let P(n) be the number of such strings of length n.
4. We notice that a length-n string can be constructed from any string of length n-1 by appending 0, or from any string of length n-2 by appending 01, and these two constructions do not share any common string due to the different last digit.
5. Thus, P(n)=P(n-1)+P(n-2).
6. We have P(1)=2 (0 and 1) and P(2)=3 (00, 01 and 10).
7. Thus, P(n) for n from 2 to 10 is 5, 8, 13, 21, 34, 55, 89 and 144.
8. So there are 144 such subsets.

Question 6:
In how many ways can you take an odd number of objects from n (distinct) objects?
Answer 6:
The process of solving the problem can be divided into the following steps:
1. Let U be the set of objects.
2. For a specific object m in the collection, we can partition the set of all subsets into two sets, A={{S\u2286U: m\u2208S}} and B={{S\u2286U: m\u2209S}}.
3. We can construct a one-to-one mapping between A and B by pairing up sets that only differ in m, so |A|=|B|.
4. Thus, one of the sets has an odd number of objects and the other has an even number of objects.
5. Since there are 2^n subsets, there are 2^(n-1) subsets with an odd number of objects.

Question 7:
Of 3n+1 objects, n are indistinguishable, and the remaining ones are distinct. In how many ways can we choose n objects, as an expression of n?
Answer 7:
1. We have 2n+1 distinguishable objects and n indistinguishable objects.
2. Thus, to choose a collection contains k distinguishable objects, there are C(2n+1, k) ways.
3. So the total number of ways is C(2n+1, 0)+C(2n+1, 1)+...+C(2n+1, n).
4. Denote C(2n+1, k) as c_k. We have c_0+c_1+...+c_(2n)+c_(2n+1)=2^(2n+1).
5. Furthermore, we have c_0=c_(2n+1), c_1=c_(2n), ..., c_n=c_(n+1).
6. Thus, we have c_0+c_1+...+c_n=2^(2n+1)/2=4^n.

Question 8:
Along a one-way street there are n parking lots. One-by-one n cars numbered 1 to n enter the street. Each driver i heads to his favorite parking lot a_i, and, if it is free, he occupies it. Otherwise, he continues to the next free lot and occupies it. But if all succeeding lots are occupied, he leaves for good. How many sequences {{a_1, ..., a_n}} are such that every driver can park, as an expression of n?
Answer 8:
1. Make the parking lot arrangement a circle, by connecting the n-th lot to a new (n+1)-th lot, which connects back to the 1st lot.
2. Thus, every car will have somewhere to park because there are n cars in the (n+1)-lot circle.
3. In addition, satisfying the original parking configuration is equivalent to the (n+1)-th lot being empty after all cars have parked (because an occupied (n+1)-th lot means that the car on that lot would have left for good originally).
4. Let each driver choose his favorite lot number from 1 to n+1, so there are (n+1)^n sequences.
5. Due to symmetry, we can split the set of all sequences into (n+1)^(n-1) groups of n sequences each, where the parking configurations of the n sequences are cyclic shifts of others.
6. In each group, only one is valid (i.e., the one with (n+1)-th lot unoccupied).
7. Thus, we have (n+1)^(n-1) valid sequences.
'''

# 存在太多公式化推导的其实都不太好分解。
# 手写示例 from scy

def decompose_sql(clients, question, config):    
    '''champ'''

    prompt_for_decompose = f"""You are an expert on mathematics. I will now give you a math problem. Please break this math problem down into several easy-to-solve steps.

4 examples are as follows:

Question 1:
How many different ordered triples (a, b, c) of non-negative integers are there such that a+b+c=99?
Answer 1:
The process of solving the problem can be divided into the following steps:
1. What are the possible values for c given that a+b+c=99?
   A: c can range from 0 to 99.
2. For a fixed value of c, what equation must a and b satisfy?
   A: a+b=99-c.
3. For a given value of c, how many combinations of non-negative integers (a,b) satisfy the equation a+b=99-c?
   A: There are (99-c)+1 combinations for each value of c.
4. What is the total number of ordered triples (a,b,c) by summing the combinations for all possible values of c from 0 to 99?
   A: The total number is the sum of all combinations, which equals 100+99+...+1=100*101/2=5050.

Question 2: 
How many strings of length 5 using the digits {{0, 1, ..., 9}} (with leading zeros allowed) have strictly increasing digits?
Answer 2: 
The process of solving the problem can be divided into the following steps:
1. What are the requirements for a string of length 5 to have strictly increasing digits?
   A: All 5 digits in the string must be distinct and arranged in increasing order.
2. What happens if the 5 digits in the string are not distinct?
   A: If there are any duplicate digits, the string cannot have strictly increasing digits.
3. How many ways are there to choose 5 distinct digits from the set {{0,1,...,9}}?
   A: The number of ways to choose 5 distinct digits from 10 digits is given by the binomial coefficient C(10,5).
4. How many strings of length 5 have strictly increasing digits?
   A: The total number of valid strings is exactly equal to C(10,5).
   
Question 3:
Does a polyhedron exists with an odd number of faces, each face having an odd number of edges?
Answer 3:
The process of solving the problem can be divided into the following steps:
1. What is the result when we sum the edges of all faces if each face has an odd number of edges and there are an odd number of faces?
   A: The sum will be an odd number (since the sum of an odd number of odd numbers is odd).
2. How is the total sum of edges calculated when considering that each edge is shared by two faces?
   A: The total sum of edges will be twice the actual number of edges, which must be an even number.
3. Can the total sum of edges be both odd (from subproblem 1) and even (from subproblem 2) at the same time?
   A: No, it is not possible for a number to be both odd and even.
4. Does a polyhedron exist that has an odd number of faces, each with an odd number of edges?
   A: No, such a polyhedron cannot exist due to the contradiction.

Question 4:
Each of the faces of a cube is painted by a different color. How many of the colorings are distinct up to rotations?
Answer 4:
The process of solving the problem can be divided into the following steps:
1. How many distinct colorings are there if we start by painting one face and placing it on the bottom?
   A: There are two cases to consider: painting the second face on top or on the side.
2. If the second face is painted on top, how many distinct colorings are there?
   A: The four remaining side faces can be colored in 4 factorial (24) ways. However, rotating the cube around the top face produces four identical colorings. So, there are 6 distinct colorings in this case.
3. If the second face is painted on a side, how many distinct colorings are there?
   A: The side face can be positioned in one of four directions, but it fixes the cube's orientation uniquely, leaving the four remaining faces with 24 possible colorings. Thus, there are 24 distinct colorings in this case.
4. What is the total number of distinct colorings of the cube?
   A: Adding both cases, we have 6 (second face on top) + 24 (second face on the side) = 30 distinct colorings.

Now the problem is {question}, please decompose it into easy-to-solve steps like the examples.
Each step in the examples includes both the subproblem and its answer. 
But here I only need you to list the subproblems without providing an answer for each subproblem.
Please write each broken-down question step on a separate line, starting with a number.
Answer Format: 
The process of solving the problem can be divided into the following steps:
1. question step 1
2. question step 2
3. question step 3
...
"""

    Q = {
        "role": "user",
        "content": prompt_for_decompose
    }
    # Query = Example+[Q]
    Query = [Q]
    # result = askChatGPT(Query, model=config["decompose_MODEL"], temperature=1)
    result = askLLM(clients, Query, tokens_path=config['tokens_path'], model=config['decompose_MODEL'], temperature=1)
    return result


def convert_steps_to_format(decom_commands):
    # 截取“we need to know:”后的内容
    start_index = decom_commands.find("The process of solving the problem can be divided into the following steps:") + len("The process of solving the problem can be divided into the following steps:")
    subtasks_text = decom_commands[start_index:].strip()

    # 将每个子任务单独列出
    subtasks = subtasks_text.split('\n')
    subtasks = [task.strip().split('. ', 1)[-1] for task in subtasks]
    
    steps_dict = {index + 1: value for index, value in enumerate(subtasks)}
    
    return subtasks, steps_dict


def AllocateModel(clients, question, steps, config):
    # 在这里就进行分配,确定用哪个大语言模型来解决问题
    # 一次性对所有的子任务分配MODEL
    # Below are some examples of model assignments based on the difficulty of the problems themselves.
    allo_Q = [{'role':'user', 'content':f"""I am now going to solve a math problem. I have broken down the initial problem into several smaller ones and hope to solve these small problems step by step using large language models(LLMs).
Now I need to assign a LLM to each small problem, which can be one of the following: [gpt-4-turbo, gpt-4, gpt-4o-mini, gpt-3.5-turbo, llama3-70b, llama3-8b].
Their capabilities are ranked from strongest to weakest as follows: [gpt-4-turbo, gpt-4, gpt-4o-mini, gpt-3.5-turbo, llama3-70b, llama3-8b]
Their costs are ranked from highest to lowest as follows: [gpt-4, gpt-4-turbo, gpt-3.5-turbo, gpt-4o-mini, llama3-70b, llama3-8b]

Now, my original math problem is: \n{question}\n
The list of subtasks is as follows: {steps}
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
    # connect_commands = askChatGPT(connect_Q, model=config["dependencies_1_MODEL"], temperature = 1)
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
    # output = askChatGPT(Query, model=config["dependencies_2_MODEL"], temperature=1)
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
    
def solve_Sub_Question(clients, question, answerDict, int_edges, number, subtask, tokens_path, answer_MODEL, alpha):
    # question 问题字符串
    # 交待解决任务
    sys_q = f"""There is a math_problem. I need you to solve it and give an answer.
Here is the problem:\n{question}

I have broken this math problem down into several smaller problems. I will assign you sub-problems one by one, and provide the results of the previous sub-problems as a reference for your reasoning.
Please solve the problem and respond according to mathematical logic.
"""  # 系统任务信息
    
    if len(answerDict)>0:
        answersSoFar = f"""\nSo far, the answers to the resolved sub-problems are as follows: The format is Sub-problem-Id: xxx; Sub-problem: xxx; Answer: xxx."""
        for key, value in answerDict.items():
            answersSoFar += f"""\nSub-problem-Id: {key}; Sub-problem: {answerDict[key]['subtask']}; Answer: {answerDict[key]['answer']}."""
        
        predecessors = search_Predecessors(int_edges, number)
        intersection = set(answerDict.keys()).intersection(set(predecessors))
        count = len(intersection)
        if count>0:
            answersSoFar += f"""\nAmong them, sub-problems {predecessors} are directly related to this sub-problem, so please pay special attention to them."""
    
    
    subask = f"""\nThe sub-problem to solve now is xxx: {subtask}
Based on the information above, please provide a concise and clear answer"""

    if len(answerDict)>0:
        query = answersSoFar+subask
    else:
        query = subask

    Q = [{'role':'system', 'content':sys_q},
        {'role':'user', 'content':query},]
    
    # 记得调整这里的模型选择
    
    # result = askChatGPT(Q, model='gpt-3.5-turbo', temperature=1, max_tokens=300)
    result, probs = askLLM_withprob(clients, Q, tokens_path=tokens_path, model=answer_MODEL, temperature=1, max_tokens=500)
    prob_values = list(probs.values())
    return result, prob_values, Q


def frac2model(frac, thres5=[0.9, 0.8, 0.5, 0.4, 0.2]):
    # gpt-4-turbo, gpt-4, gpt-4o-mini, gpt-3.5-turbo, llama3-70b, llama3-8b
    if frac>thres5[0]:  # 0.9
        return 'llama3-8b'
    elif frac>thres5[1]:  # 0.8
        return 'llama3-70b'
    elif frac>thres5[2]:  # 0.5
        return 'gpt-3.5-turbo'
    elif frac>thres5[3]:  # 0.4
        return 'gpt-4o-mini'
    elif frac>thres5[4]:  # 0.2
        return 'gpt-4'
    else:
        return 'gpt-4-turbo'

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
    
    
    