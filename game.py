import numpy as np

board = None
steps = None

def winner():
    for row in board:
        s = row.sum()
        if s == 3:
            return 1
        elif s == (-3):
            return -1
    for column in board.transpose():
        s = column.sum()
        if s == 3:
            return 1
        elif s == (-3):
            return -1
    d1 = board.diagonal().sum()
    if d1 == 3:
        return 1
    elif d1 == (-3):
        return -1
    d2 = np.fliplr(board).diagonal().sum()
    if d2 == 3:
        return 1
    elif d2 == (-3):
        return -1
    return 0

def active_player():
    out = winner()
    if out:
        return out * 10
    s = board.sum()
    if s == 1:
        return -1
    else:
        return 1

def new_game():
    global board
    global steps
    board = np.zeros((3, 3))
    steps = 0

def step(index, player):
    global steps
    reward = -0.1
    (x, y) = (index // 3, index % 3)
    if board[x, y] == 0:
        steps = steps + 1
        board[x, y] = player
    else:
        reward = reward - 10
    a_p = active_player()
    if abs(a_p) == 10:
        reward = reward + 1
    return a_p, reward


# board = np.array([[1,1,-1], [0,-1,0], [-1,0,0]])
new_game()
print(step(2, 1))
print(board)
print(step(2, 1))