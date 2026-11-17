class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next


def rotate_right(head: Node, k: int) -> Node:
    if not head or not head.next or k == 0:
        return head

    # 1. Находим длину списка и его текущий хвост
    length = 1
    tail = head
    while tail.next:
        tail = tail.next
        length += 1

    # 2. Убираем лишние полные обороты
    k %= length
    if k == 0:
        return head

    # 3. Замыкаем список в кольцо
    tail.next = head

    # 4. Ищем новый хвост (он находится на позиции length - k - 1 от начала)
    steps_to_new_tail = length - k - 1
    new_tail = head
    for _ in range(steps_to_new_tail):
        new_tail = new_tail.next

    # 5. Определяем новую голову и разрываем кольцо
    new_head = new_tail.next
    new_tail.next = None

    return new_head
