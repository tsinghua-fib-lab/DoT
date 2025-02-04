import sys
from collections import defaultdict, deque

sys.path.append('C:\\Users\\Pluto\\Desktop\\TaDe')
import re
from typing import Any, List

from utils import *

GPT_MODEL = "gpt-4-turbo"

def extract_in_out(text):
    # 使用正则表达式提取IN和OUT后的内容
    match = re.search(r'IN:\s*(.*?)\s*OUT:\s*(.*)', text)
    question = match.group(1)
    actions = match.group(2)
    return question, actions
    

def decompose_sql(question):    
    
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
#     answer_example = f""" a valid 'start'. ]
# STEP6 [ For each 'start' value in the loop, calculate sum(seq[start:start + k]). ]
# STEP1 [ Understand the puzzle's primary objective: Find an integer 'start' such that when added to the subsequent 'k' elements in 'seq', the sum is greater than or equal to 'lower'. ]
# STEP2 [ Note the inputs: 'k', 'lower', and 'seq' are given and fixed. ]
# STEP3 [ Recognize the constraints: The variable 'start' must be within the range 0 to len(seq) - k. ]
# STEP4 [ Comprehend that sum(seq[start:start + k]) needs to be greater than or equal to 'lower'. ]
# STEP5 [ Deduce a strategy: Loop through the sequence from 0 to len(seq) - k to find
# STEP7 [ Compare the calculated sum with 'lower'. ]
# STEP8 [ If the sum is greater than or equal to 'lower', return that 'start' value. ]
# """

#     Example = [
#         {"role": "user", "content": question_example},
#         {"role": "assistant", "content": answer_example}
#     ]


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
    result = askChatGPT(Query, model=GPT_MODEL, temperature=1)
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


import ipydagred3
import networkx as nx


def construct_dependencies_without_traversal(question, steps):
    # 这里的steps只是一个list
    
    connect_system = {"role": "system", "content": """Now we have a natural language instruction, which we have broken down into many subcommands in the form of a list. I want you to understand the connection between these subcommands."""}
    connect_user = {"role": "user", "content":f"The init natural language instruction is {question}. And the subcommands are {steps}. Please provide your understanding of the relationships between these subcommands."}
    connect_Q = [connect_system, connect_user]
    connect_commands = askChatGPT(connect_Q, model=GPT_MODEL, temperature = 1)
    # print('LLM对于subcommands之间关系的语言理解:\n\n', connect_commands)

    prompt = f"""Now we need to create standardized connections for the relationships between subcommands.
    Now Given the following subtasks for question: {question}, determine the dependencies between them:\n"""

    for count, step in enumerate(steps, start=1):
        prompt += f"Step {count}: {step}\n"
        
    prompt += """\nPlease list the dependencies in the format 'Subtask A [xxx] -> Subtask B [xxx]' indicating that Subtask A must be completed before Subtask B can start.
Please identify any potential conditional dependencies from a logical perspective.

Here are some examples:
"jump right" -> "jump opposite right"
"jump opposite right" -> "jump opposite right twice"
"walk around left" -> "walk around left thrice"
"walk left" -> "walk around left"
"run around left" -> "run around left thrice"
"run around right" -> "run around right twice"

Answer format: (Please strictly follow the format. Each dependency should be separated by a new line. No explanation is required.)
Step id_i [ subtask i ] -> Step id_j [ subtask j ]
Step id_m [ subtask m ] -> Step id_n [ subtask n ]
...
"""
    prompt_user = {
        "role": "user",
        "content": f"""{prompt}"""
    }
    Query = [prompt_user]
    output = askChatGPT(Query, model=GPT_MODEL, temperature=1)
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
    
    
    