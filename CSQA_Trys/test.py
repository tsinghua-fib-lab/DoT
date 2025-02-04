import json
import random
import sys

sys.path.append('C:\\Users\\Pluto\\Desktop\\TaDe')
from utils import *

# 假设 JSON 数据保存在名为 'data.json' 的文件中
with open('QA-1008_CSQA_Dataset_4finetuning.json', 'r') as file:
    data = json.load(file)

diff = []
count = 0

# 打印每个问题的相关信息
for item in data:
    # print("Problem Text:", item['problemText'].strip())
    # print("All Subtasks:", item['allSubtask'].strip())
    # print("Current Subtask:", item['nowSubtask'].strip())
    # print("Difficulty Number:", item['difficultyNum'])
    # print("-" * 40)  # 分隔线
    diff.append(item['difficultyNum'])

   
                
lst = diff
count_0 = lst.count(0)
count_1 = lst.count(1)
print(f"小模型占比为: {count_0/(count_1+count_0)}")


