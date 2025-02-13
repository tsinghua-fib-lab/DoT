import json
import random

def merge_and_shuffle_json(file1, file2, file3, output_file):
    # 读取第一个文件的list
    with open(file1, 'r') as f1:
        list1 = json.load(f1)
    
    # 读取第二个文件的list
    with open(file2, 'r') as f2:
        list2 = json.load(f2)
    
    # 读取第三个文件的list
    with open(file3, 'r') as f3:
        list3 = json.load(f3)
    
    # 合并三个文件的list
    combined_list = list1 + list2 + list3
    
    # 随机打乱合并后的list
    random.shuffle(combined_list)
    
    # 统计difficultyNum的数量
    count_1 = sum(1 for item in combined_list if item.get('difficultyNum') == 1)
    count_0 = sum(1 for item in combined_list if item.get('difficultyNum') == 0)
    
    # 输出统计结果
    print(f"'difficultyNum' == 1: {count_1}")  # 1有1057
    print(f"'difficultyNum' == 0: {count_0}")  # 0有9245
    
    # 将打乱后的list写入新的文件
    with open(output_file, 'w') as out_file:
        json.dump(combined_list, out_file, indent=4)
    
    print(f"len(combined_list): {len(combined_list)}")
    print(f"Successfully merged and shuffled lists from {file1}, {file2}, {file3} into {output_file}")

# 指定三个文件的路径
file1 = '../CHAMP/QA-1008_CHAMP_Dataset_4finetuning.json'
file2 = '../MATH/QA-1008_MATH_Dataset_4finetuning.json'
file3 = '../DROP/QA-1008_DROP_Dataset_4finetuning.json'
output_file = 'QA-1008_mathmatics3_Dataset_4finetuning.json'

# 调用函数进行合并和打乱
merge_and_shuffle_json(file1, file2, file3, output_file)
