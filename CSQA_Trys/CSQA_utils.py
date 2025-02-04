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
    
# 问题分解prompts设计完毕
def decompose_sql(clients, question, options, config):   
    '''
    1 examples are as follows:
    Question: 
    '''
    # options已经由Dict转化成了String的形式.
    prompt_for_decompose = f"""I have a single-choice question involving common sense reasoning that I want to solve. I hope you can break down the problem-solving process into several sub-problems. You can consider analyzing the question itself as well as the options.
The number of sub-problems doesn't need to be too many; each sub-problem should have a clear meaning and purpose.

8 examples are as follows:
Question 1:
You can read a magazine where while waiting for your transportation on rails to arrive?
Choices 1: 
A. Train station, B. Bookstore, C. Newsstand, D. Waiting room, E. Airport
Answer 1:
The solution to this problem can be approached through the following steps:  
1. What does "waiting for your transportation on rails" indicate about your current location?
2. Which place in the options can accommodate you reading a magazine?
3. Which places that satisfy question 2 are near your current location?

Question 2:
If I wanted to see a lizard in its natural habitat but I do not speak Spanish, where would I go?
Choices 2: 
A. Utahc, B. South America, C. New Hampshire, D. Japan, E. New Mexico
Answer 2:
1. Which places are natural habitats for lizards?
2. Which places have Spanish as the primary language?
3. Combine the answers from sub-question 1 and sub-question 2, among the natural habitats for lizards, which places do not speak Spanish?

Question 3:
John was stuck in his house.  He couldn't get out the door.  He was very frightened when the smoke detectors went off, but luckily it was a false alarm.  Why might he be stuck?
Choices 3:
A. fire,  B. belong to, C. winter storm, D.face south, E.burn down
Answer 3:
1. What are possible reasons for being stuck in a house?
2. Which options are related to situations that might cause a person to be stuck?
3. Why might these specific conditions make it difficult to leave the house?

Question 4:
John was stuck in his house.  He couldn't get out the door.  He was very frightened when the smoke detectors went off, but luckily it was a false alarm.  Why might he be stuck?
Choices 4:
A. fire,  B. belong to, C. winter storm, D.face south, E.burn down
Answer 4:
1. What are possible reasons for being stuck in a house?
2. Which options are related to situations that might cause a person to be stuck?
3. Why might these specific conditions make it difficult to leave the house?

Question 5:
When looking for a non-perishable food in your house, you'll often go look in the?
Choices 5:
A. Stove, B. Table, C. Plate, D. Jar, E. Pantry
Answer 5:
1. What is non-perishable food?
2. Where are non-perishable foods commonly stored in a household?
3. Which of the options (stove, table, plate, jar, pantry) is the most logical place for storing non-perishable food?

Question 6:
What must elementary school students do when they are going into class?
Choices 6:
A. Think for himself, B. Answer question, C. Wait in line, D. Speak a foreign language, E. Cross road
Answer 6:
1. What do elementary school students typically do before entering a classroom?
2. Which actions among the options are related to classroom entry procedures?
3. Why might students perform this action before entering the classroom?

Question 7:
After eating dinner, having plenty to eat, and exercising, what is likely to happen?
Choices 7:
A. Become tired, B. Indigestion, C. Flatulence, D. Become intoxicated, E. Become full
Answer 7:
1. What happens to the body after eating a large meal?
2. What are common effects of exercising after eating?
3. Which of the options (become tired, indigestion, flatulence, become intoxicated, become full) best matches the expected outcome of eating a large meal followed by exercise?

Question 8:
He didn't like the walk up, but living on the top floor meant there was nobody above him in the what?
Choices 8:
A. Apartment building, B. Tall building, C. Go down, D. Garden, E. Office building
Answer 8:
1. What does "walk up" suggest about the type of building?
2. What kind of building would have a "top floor" and residents living above or below each other?
3. Which option (apartment building, tall building, go down, garden, office building) best describes a place where living on the top floor would mean no one lives above?


Now the question is {question}, the options are: {options}, please decompose it into sub-questions. 
Answer Format: (Please write each broken-down question on a separate line, starting with a number.)
To solve the question "xxx", we need to clarify / solve:
"1. sub-question 1",
"2. sub-question 2",
"3. sub-question 3".
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
    start_index = decom_commands.find("clarify / solve:") + len("clarify / solve:")
    subtasks_text = decom_commands[start_index:].strip()

    # 将每个子任务单独列出
    subtasks = subtasks_text.split('\n')
    subtasks = [item for item in subtasks if item != '']  # 有的时候会有空字符串，需要删掉
    subtasks = [task.strip().split('. ', 1)[-1] for task in subtasks]
    
    steps_dict = {index + 1: value for index, value in enumerate(subtasks)}
    
    return subtasks, steps_dict




def construct_dependencies_without_traversal(clients, question, options_string, steps, config):
    # 这里的steps只是一个list
    
    connect_system = {"role": "system", "content": """Now we have a single-choice question involving common sense reasoning, which we have broken down into many sub-questions. I want you to understand the connection between these sub-questions."""}
    connect_user = {"role": "user", "content":f"The question is {question}. The options are {options_string}. And the sub-questions are {steps}. Please provide your understanding of the relationships between these sub-questions. Your response must be concise."}
    connect_Q = [connect_system, connect_user]
    # connect_commands = askChatGPT(connect_Q, model=GPT_MODEL, temperature = 1)
    connect_commands = askLLM(clients, connect_Q, tokens_path=config['tokens_path'], model=config['dependencies_1_MODEL'], temperature=1)
    # print('LLM对于subcommands之间关系的语言理解:\n\n', connect_commands)
    connect_Q.append({
        "role": "assistant",
        "content": connect_commands,
    })

    prompt = f"""Now we need to create standardized connections for the relationships between these sub-questions.
    Now given the following sub-questions for the question: {question} and options: {options_string}, determine the dependencies between them:\n"""

    for count, step in enumerate(steps, start=1):
        prompt += f"Step {count}: {step}\n"
        
    prompt += """\nPlease list the dependencies in the format 'Sub-question A [xxx] -> Sub-question B [xxx]' indicating that Sub-question A must be completed before Sub-question B can start.
Please identify any potential conditional dependencies from a logical perspective.

Answer format: (Please strictly follow the format. Each dependency should be separated by a new line. No explanation is required.)
Step id_i [ sub-question i ] -> Step id_j [ sub-question j ]
Step id_m [ sub-question m ] -> Step id_n [ sub-question n ]
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
    
    


def AllocateModel(clients, question, options, steps, config):
    # 在这里就进行分配,确定用哪个大语言模型来解决问题
    # 一次性对所有的子任务分配MODEL
    # Below are some examples of model assignments based on the difficulty of the problems themselves.
    allo_Q = [{'role':'user', 'content':f"""I am now going to solve a single-choice question involving common sense reasoning. I have broken down the initial question into several smaller ones and hope to solve these small questions step by step using large language models(LLMs).
Now I need to assign a LLM to each small question, which can be one of the following: [gpt-4-turbo, gpt-4, gpt-4o-mini, gpt-3.5-turbo, llama3-70b, llama3-8b].
Their capabilities are ranked from strongest to weakest as follows: [gpt-4-turbo, gpt-4, gpt-4o-mini, gpt-3.5-turbo, llama3-70b, llama3-8b]
Their costs are ranked from highest to lowest as follows: [gpt-4, gpt-4-turbo, gpt-3.5-turbo, gpt-4o-mini, llama3-70b, llama3-8b]

Now, my original question is: \n{question}\n
The options are: \n{options}\n
The list of the sub-questions is as follows: \n{steps}\n
Please help me evaluate the difficulty of each sub-question (using an integer score from 1 to 5) and assign a dedicated MODEL to each.
Output format: (No additional explanations or comments are needed. Please strictly follow the format below in your response. Place the content, difficulty, and assigned model of each sub-question in a tuple, and put all the tuples in a list.) 
[
    ("subquestion1", "difficulty1", "MODEL1"),
    ("subquestion2", "difficulty2", "MODEL2"),
    ("subquestion3", "difficulty3", "MODEL3"),
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


def solve_Sub_Question(clients, answer_MODEL, question, options_string, answerDict, int_edges, number, subtask, tokens_path, alpha):
    # question 问题字符串
    # 交待解决任务
    # 主要是answer_MODEL可能会变
    
    sys_q = f"""There is a single-choice question involving common sense reasoning. I need you to solve it and give the right answer.
Here is the question:\n{question} 
Here are the options: \n{options_string}

I have broken this common sense reasoning question down into several smaller questions. I will assign you sub-questions one by one, and provide the results of the previous sub-questions as a reference for your reasoning."""  # 系统任务信息
    
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
    
    
    