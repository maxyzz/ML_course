# a = []
# b = [1, 2, 3]


def merge(a, b):
    result = []
    i = 0
    j = 0

    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1

    while i < len(a):
        result.append(a[i])
        i += 1

    while j < len(b):
        result.append(b[j])
        j += 1

    return result

# print(merge(a, b))  # Output: [1, 2, 3, 5, 6, 7, 8, 13]