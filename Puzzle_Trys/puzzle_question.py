'''
just检测问题有没有回答正确
'''

from typing import List


def sat(val_index: List[int], nums=[4512821, 7022753, 5506558]):
    if val_index == []:
        return all(n % 2 == 1 for n in nums)
    v, i = val_index
    assert v % 2 == 0 and nums[i] == v
    return all(n > v or n % 2 == 1 for n in nums[:i]) and all(n >= v or n % 2 == 1 for n in nums[i:])

print(sat([5506558, 2]))


