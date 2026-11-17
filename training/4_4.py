class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def kth_smallest(head: Node, k: int) -> int:
    stack = []
    curr = head

    while curr or stack:
        # 1. Идем максимально влево, сохраняя путь в стек
        while curr:
            stack.append(curr)
            curr = curr.left

        # 2. Извлекаем очередной наименьший элемент
        curr = stack.pop()
        k -= 1

        # 3. Если счетчик обнулился, нужный элемент найден
        if k == 0:
            return curr.val

        # 4. Переходим к правому поддереву
        curr = curr.right

    return -1  # На случай, если k больше числа узлов
