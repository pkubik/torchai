"""
Cart Pole problem solved using simple DQN with replay buffer.

Inspired by end-to-end tutorial for screen to action mapping using matplotlib:
http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
(although only ``gym`` observations are being used in this script)
"""

import gym
import matplotlib.pyplot as plt
from typing import Iterable
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# if gpu is to be used
from torchai.utils import Transition, ReplayMemory, DecayingBinaryRandom

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


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
        self.activation = nn.ReLU()

    def forward(self, x):
        return nn.Sequential(
            self.fc1,
            self.bn1,
            self.activation,
            self.fc2,
            self.bn2,
            self.activation,
            self.fc3,
            self.bn3,
            self.activation,
            self.fc4
        )(x)


class Batch:
    def __init__(self, data):
        self.state = [record.state for record in data]
        self.state_tensor = FloatTensor(self.state)
        self.action = [int(record.action) for record in data]
        self.action_tensor = LongTensor(self.action)
        self.reward = [record.reward for record in data]
        self.reward_tensor = FloatTensor(self.reward)
        self.next_state = [record.next_state for record in data]
        self.next_state_tensor = FloatTensor([s for s in self.next_state if s is not None])


class Actor:
    def __init__(self):
        self.decaying_binary_random = DecayingBinaryRandom(eps_end=0.01)
        self.gamma = 0.999
        self.model = DQN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.response_counter = 0

    def apply_noise_to_response(self, response):
        if self.decaying_binary_random.sample(decay_steps=self.response_counter):
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
        batch = Batch(data)
        state_var = Variable(batch.state_tensor)
        action_var = Variable(batch.action_tensor)
        reward_var = Variable(batch.reward_tensor)

        non_final_mask = ByteTensor([s is not None for s in batch.next_state])
        non_final_next_states = Variable(batch.next_state_tensor, volatile=True)

        action_values = self.model.train()(state_var).gather(1, action_var.unsqueeze(-1))

        next_state_values = Variable(torch.zeros(state_var.data.size(0)).type(Tensor))
        next_state_values[non_final_mask] = self.model.eval()(non_final_next_states).max(1)[0]
        expected_action_values = (self.gamma * next_state_values) + reward_var

        # Huber loss
        loss = F.smooth_l1_loss(action_values, expected_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def run():
    env = gym.make('CartPole-v0').unwrapped
    plt.ion()

    batch_size = 100
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
                    t, actor.decaying_binary_random.eps_threshold(actor.response_counter)))
                break

    print('Complete')
    env.close()
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run()
