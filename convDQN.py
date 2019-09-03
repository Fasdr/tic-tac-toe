import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import game

conv_size = 20
reshape_size = conv_size * 3
input_layer_size = conv_size * 8

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, conv_size, (3, 1))
        self.conv2 = nn.Conv2d(1, conv_size, (1, 3))
        self.lin1 = nn.Linear(3, conv_size)
        self.lin2 = nn.Linear(3, conv_size)
        self.lin5 = nn.Linear(input_layer_size, input_layer_size)
        self.lin3 = nn.Linear(input_layer_size, 9)
        self.lin4 = nn.Linear(input_layer_size, 9)
    def forward(self, x):
        y = x.reshape(1, 1, 3, 3)
        out1 = self.conv1(y)
        out2 = self.conv2(y)
        out1 = out1.reshape(reshape_size)
        out2 = out2.reshape(reshape_size)
        d1 = x.diagonal()
        d2 = x.rot90(1, (1, 0)).diagonal()
        out3 = self.lin1(d1)
        out4 = self.lin1(d2)
        x = x.reshape(9)
        mask1 = torch.where(x == 0, x * 0, x * 0 + 1)
        mask2 = 1 - mask1
        mask1 = mask1
        pred = F.relu(torch.cat((out1, out2, out3, out4), dim=0))
        pred = F.relu(self.lin5(pred))
        fout1 = F.relu(self.lin3(pred))
        fout2 = F.relu(self.lin4(pred))
        return mask2*fout1 - mask1, mask2*fout2 - mask1

GAMMA = 0.7
ALPHA = 0.001
model = DQN()
optimizer = optim.SGD(model.parameters(), lr=ALPHA)
loss_for_one_episode = 0
loss_for_sever_episodes = 0
in_eps = 1/2
agr = 1/20
runs = 100000
with_graph = True
graph = []
big_step = 100

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
        loss = ((reward - q_value)**2)
    else:
        while a_p != player and abs(a_p) != 10 and not game.full_board():
            index = epsilon_greedy(agr, a_p)
            a_p, _ = game.step(index, a_p)
        if abs(a_p) == 10:
            loss = ((reward - 17 - q_value)**2)
        elif game.full_board():
            loss = ((reward - 5 - q_value)**2)
        else:
            q_value_max = (model(torch.FloatTensor(game.board) * player)[(a_p+2)%3]).max()
            loss = ((reward + GAMMA*q_value_max - q_value)**2)
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
    global loss_for_one_episode, loss_for_sever_episodes
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
    loss_for_sever_episodes += loss_for_one_episode


for i in range(1, runs+1):
    one_episode(in_eps, i%2)
    if i%big_step == 0:
        print(i)
        graph.append(loss_for_sever_episodes/big_step)
        # print(loss_for_sever_episodes/big_step)
        loss_for_sever_episodes = 0

if not with_graph:
    while True:
        play_with()
if with_graph:
    plt.plot(range(10, runs+1, big_step), graph)
    plt.axis([0, runs+1, 0, 500])
    plt.show()
    print(len(graph))
