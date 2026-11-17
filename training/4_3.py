def drown(board: list[list[str]]) -> list[list[str]]:
    if not board or not board[0]:
        return board

    rows, cols = len(board), len(board[0])

    def dfs(r: int, c: int) -> None:
        # Выходим за пределы поля или упираемся не в 'O'
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != 'O':
            return
        
        # Помечаем безопасную клетку временным символом
        board[r][c] = 'T'
        
        # Проверяем 4 соседних направления
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    # 1. Запускаем обход от всех 'O' на левой и правой границах
    for r in range(rows):
        if board[r][0] == 'O':
            dfs(r, 0)
        if board[r][cols - 1] == 'O':
            dfs(r, cols - 1)

    # 2. Запускаем обход от всех 'O' на верхней и нижней границах
    for c in range(cols):
        if board[0][c] == 'O':
            dfs(0, c)
        if board[rows - 1][c] == 'O':
            dfs(rows - 1, c)

    # 3. Затапливаем оставшиеся 'O' и восстанавливаем безопасные клетки 'T'
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 'O':
                board[r][c] = 'X'
            elif board[r][c] == 'T':
                board[r][c] = 'O'

    return board
