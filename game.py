import numpy as np

board = None
def full_board():
    return np.where(board == 0, 0, 1).sum() == 9

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
    board = np.zeros((3, 3))

def step(index, player):
    reward = -1
    (x, y) = (index // 3, index % 3)
    board[x, y] = player
    a_p = active_player()
    if abs(a_p) == 10:
        reward = reward + 40
    elif full_board():
        reward = reward + 20
    return a_p, reward


# board = np.array([[1,1,-1], [0,-1,0], [-1,0,0]])
# new_game()
# print(step(2, 1))
# print(board)
# print(step(2, 1))