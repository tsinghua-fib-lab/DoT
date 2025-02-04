import json
import random
import sys

sys.path.append('C:\\Users\\Pluto\\Desktop\\TaDe')
from utils import *

# 假设 JSON 数据保存在名为 'data.json' 的文件中
with open('step1Res_MATH.json', 'r') as file:
    data = json.load(file)

diff = []
count = 0

# 打印每个问题的相关信息
for item in data.values():
    # print("Problem Text:", item['problemText'].strip())
    # print("All Subtasks:", item['allSubtask'].strip())
    # print("Current Subtask:", item['nowSubtask'].strip())
    # print("Difficulty Number:", item['difficultyNum'])
    # print("-" * 40)  # 分隔线
    diff.append(len(item["steps"]))

   
                
print(sum(diff)/len(diff))

