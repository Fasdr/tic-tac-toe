import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import game

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, (3, 1))
        self.conv2 = nn.Conv2d(1, 30, (1, 3))
        self.lin1 = nn.Linear(3, 30)
        self.lin2 = nn.Linear(3, 30)
        # 30*3+30*3+30+30
        self.lin3 = nn.Linear(240, 9)
        self.lin4 = nn.Linear(240, 9)
    def forward(self, x):
        y = x.reshape(1, 1, 3, 3)
        out1 = self.conv1(y)
        out2 = self.conv2(y)
        out1 = out1.view(-1, self.num_flat_features(out1))
        out2 = out2.view(-1, self.num_flat_features(out2))
        d1 = x.diagonal()
        d2 = x.rot90(1, (1, 0)).diagonal()
        out3 = self.lin1(d1)
        out4 = self.lin1(d2)
        x = x.reshape(9)
        mask1 = torch.where(x == 0, x * 0, x * 0 + 1)
        mask2 = 1 - mask1
        mask1 = mask1
        pred = F.relu(torch.cat((out1, out2, out3, out4), dim=0))
        fout1 = F.relu(self.lin3(pred))
        fout2 = F.relu(self.lin4(pred))
        return mask2*fout1 - mask1, mask2*fout2 - mask1
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

GAMMA = 0.75
ALPHA = 0.0005
model = DQN()
optimizer = optim.SGD(model.parameters(), lr=ALPHA)
loss_for_one_episode = 0

def epsilon_greedy(epsilon, player):
    if np.random.rand(1).item() <= epsilon:
        mask = np.where(game.board == 0, 1, 0)
        index = (np.random.rand(3, 3) * mask).argmax()
    else:
        indices = model(torch.FloatTensor(game.board))[(player+2)%3]
        index = torch.argmax(indices)
    return index

def deep_q_learning_step(epsilon, player):
    global loss_for_one_episode
    index = epsilon_greedy(epsilon, player)
    q_value = (model(torch.FloatTensor(game.board))[(player+2)%3])[index]
    a_p, reward = game.step(index, player)
    if abs(a_p) == 10 or game.full_board():
        loss = torch.abs(reward - q_value)
    else:
        while a_p != player and abs(a_p) != 10 and not game.full_board():
            index = epsilon_greedy(1/10, a_p)
            a_p, _ = game.step(index, a_p)
        if abs(a_p) == 10:
            loss = torch.abs(reward - 15 - q_value)
        elif game.full_board():
            loss = torch.abs(reward + 6 - q_value)
        else:
            q_value_max = (model(torch.FloatTensor(game.board) * player)[(a_p+2)%3]).max()
            loss = torch.abs(reward + GAMMA*q_value_max - q_value)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_for_one_episode = loss_for_one_episode + loss

    return a_p

def play_with():
    game.new_game()
    player = 1
    print(game.board)
    while abs(player) != 10 and not game.full_board():
        index = epsilon_greedy(0.0, player)
        player, _ = game.step(index, player)
        print(game.board)
        if not (abs(player) != 10 and not game.full_board()):
            continue
        my_index = -1 + int(input("index: "))
        player, _ = game.step(my_index, player)
        print(game.board)

def test():
    game.new_game()
    player = 1
    print(game.board)
    while abs(player) != 10 and not game.full_board():
        index = epsilon_greedy(0.0, player)
        player, _ = game.step(index, player)
        print(game.board)

def one_episode(epsilon, player):
    game.new_game()
    global loss_for_one_episode
    loss_for_one_episode = 0
    if player == 1:
        while abs(player) != 10 and not game.full_board():
            player = deep_q_learning_step(epsilon, player)
    else:
        index = epsilon_greedy(0.0, 1)
        player, _ = game.step(index, player)
        while abs(player) != 10 and not game.full_board():
            player = deep_q_learning_step(epsilon, player)
    print(loss_for_one_episode)

for i in range(10):
    print(i)
    one_episode(1/2, i%2)

while True:
    play_with()