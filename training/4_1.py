class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def are_trees_equal(head_1: Node, head_2: Node) -> bool:
    # 1. Если оба узла пустые, поддеревья одинаковы
    if not head_1 and not head_2:
        return True

    # 2. Если один узел пустой, а другой нет, структура отличается
    if not head_1 or not head_2:
        return False

    # 3. Если значения в текущих узлах не совпадают, деревья разные
    if head_1.val != head_2.val:
        return False

    # 4. Рекурсивно проверяем левые и правые поддеревья
    return are_trees_equal(head_1.left, head_2.left) and are_trees_equal(head_1.right, head_2.right)
