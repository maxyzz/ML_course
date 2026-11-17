# nums = [3,5,5,1]

def maj_element(nums):
    count = {}
    for num in nums:
        count[num] = count.get(num, 0) + 1
    for num, freq in count.items():
        if freq >= len(nums) // 2:
            return num
    return None

# print(maj_element(nums))  