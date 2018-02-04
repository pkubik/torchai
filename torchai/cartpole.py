"""
Cart Pole problem solved using simple DQN with replay buffer.

Inspired by end-to-end tutorial for screen to action mapping using matplotlib:
http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
(although only ``gym`` observations are being used in this script)
"""

import gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from typing import Iterable
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, record: Transition):
        self.buffer.append(record)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    fc1_size = 100
    fc2_size = 100
    fc3_size = 100

    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, self.fc1_size)
        self.bn1 = nn.BatchNorm1d(self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.bn2 = nn.BatchNorm1d(self.fc2_size)
        self.fc3 = nn.Linear(self.fc2_size, self.fc3_size)
        self.bn3 = nn.BatchNorm1d(self.fc3_size)
        self.fc4 = nn.Linear(self.fc3_size, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.fc4(x)
        return x


class Actor:
    def __init__(self):
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 1000
        self.gamma = 0.999
        self.model = DQN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.response_counter = 0

    def get_epsilon_threshold(self):
        decay_factor = math.exp(-1. * self.response_counter / self.eps_decay)
        return self.eps_end + (self.eps_start - self.eps_end) * decay_factor

    def apply_noise_to_response(self, response):
        eps_threshold = self.get_epsilon_threshold()
        if random.random() < eps_threshold:
            _, index = torch.randn(2).max(0)
            return index[0]
        else:
            return response

    def respond(self, state):
        self.response_counter += 1
        state_variable = Variable(FloatTensor([state]), volatile=True)
        response = self.model.eval()(state_variable).max(1)[1]
        return response.cpu().data[0]

    def optimization_step(self, data: Iterable[Transition]):
        state_batch = Variable(FloatTensor([record.state for record in data]))
        action_batch = Variable(LongTensor([int(record.action) for record in data]))
        reward_batch = Variable(FloatTensor([record.reward for record in data]))

        non_final_mask = ByteTensor([record.next_state is not None for record in data])
        non_final_next_states = Variable(FloatTensor([record.next_state for record in data
                                                      if record.next_state is not None]),
                                         volatile=True)

        action_values = self.model.train()(state_batch).gather(1, action_batch.unsqueeze(-1))

        next_state_values = Variable(torch.zeros(state_batch.data.size(0)).type(Tensor))
        next_state_values[non_final_mask] = self.model.train()(non_final_next_states).max(1)[0]
        next_state_values.volatile = False
        expected_action_values = (self.gamma * next_state_values) + reward_batch

        # Huber loss
        loss = F.smooth_l1_loss(action_values, expected_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def run():
    env = gym.make('CartPole-v0').unwrapped
    plt.ion()

    batch_size = 500
    actor = Actor()

    if use_cuda:
        actor.model.cuda()

    memory = ReplayMemory(10000)

    num_episodes = 100000
    for i_episode in range(num_episodes):
        state = env.reset()
        for t in count():
            original_response = actor.respond(state)
            noised_response = actor.apply_noise_to_response(original_response)
            action = noised_response

            next_state, reward, done, _ = env.step(action)
            env.render()

            if done:
                next_state = None
            memory.push(Transition(state, action, next_state, reward))
            state = next_state

            if len(memory) < batch_size:
                break

            batch = memory.sample(batch_size)

            actor.optimization_step(batch)
            if done:
                print("Episode duration: {} | Epsilon threshold: {}".format(
                    t, actor.get_epsilon_threshold()))
                break

    print('Complete')
    env.close()
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run()
