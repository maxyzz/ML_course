def color_sort(nums):
    for i in range(len(nums)-1):
         for j in range(len(nums)-1-i):
            if nums[j]>nums[j+1]:
                tmp = nums[j+1]
                nums[j+1] = nums[j]
                nums[j] = tmp
    return nums

# nums = [2, 0, 1, 0, 2]
# print(color_sort(nums))