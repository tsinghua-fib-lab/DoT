'''
其实只是为了看看puzzle的具体内容
'''

import json

filename = 'choose.txt'

with open(filename, 'r') as file:
    lines = file.readlines()

# 去除行尾的换行符并将内容存储到列表中
task_list = [line.strip() for line in lines]
task_list = ['EvenOddSum:4']


with open("puzzles.json", "r") as f:
    puzzles = json.load(f)

ids = []
for i in range(len(puzzles)):
    if puzzles[i]['name'] in task_list:
        ids.append(i)

print(ids)



# import json

# with open("puzzles.json", "r") as f:
#     puzzles = json.load(f)

# types = []
# for i in range(len(puzzles)):
#     if puzzles[i]["ans_type"] not in types:
#         types.append(puzzles[i]["ans_type"])
# print(types) 

# ['str', 'List[int]', 'int', 'List[str]', 'float', 'List[List[int]]', 'List[List[List[int]]]', 'List[float]', 'List[bool]', 'List[List[float]]', 'List[List[str]]', 'bool']


