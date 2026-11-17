class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def invert_tree(head: Node) -> Node:
    if not head:
        return None

    # 1. Рекурсивно инвертируем левое и правое поддеревья
    head.left, head.right = invert_tree(head.right), invert_tree(head.left)

    return head
