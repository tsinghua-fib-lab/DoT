import json
import random
import sys

sys.path.append('C:\\Users\\Pluto\\Desktop\\TaDe')
from utils import *

# 假设 JSON 数据保存在名为 'data.json' 的文件中
with open('QA-plus_MATH_Dataset_4finetuning.json', 'r') as file:
    data = json.load(file)

diff = []
count = 0
countx = 0
county = 0

# 打印每个问题的相关信息
for item in data:
    # print("Problem Text:", item['problemText'].strip())
    # print("All Subtasks:", item['allSubtask'].strip())
    # print("Current Subtask:", item['nowSubtask'].strip())
    # print("Difficulty Number:", item['difficultyNum'])
    # print("-" * 40)  # 分隔线
    diff.append(item['difficultyNum'])
                
                
count_equal_0 = sum(1 for x in diff if x == 0)
count_equal_0 = sum(1 for x in diff  if x > 0)


print(f"Number of 0s: {count_equal_0}")
print(f"Number of >0s: {count_equal_0}")

print(f"小模型的占比为: {count_equal_0/(count_equal_0+count_equal_0)}")


