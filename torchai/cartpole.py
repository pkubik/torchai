"""
Cart Pole problem solved using simple DQN with replay buffer.

Inspired by end-to-end tutorial for screen to action mapping using matplotlib:
http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
(although only ``gym`` observations are being used in this script)
"""
from collections import deque

import gym
import matplotlib.pyplot as plt
from typing import Iterable
from itertools import count

import numpy as np
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
    input_size = 4
    output_size = 2
    fc1_size = 4
    #fc2_size = 32
    num_capsules = 10
    attention_softening_factor = [0 for _ in range(num_capsules)]

    def __init__(self, use_symmetry=False):
        super(DQN, self).__init__()
        self.use_symmetry = use_symmetry

        self.cou = 0

        self.bn0 = nn.BatchNorm1d(self.input_size)
        self.fc1 = nn.Linear(self.input_size, self.fc1_size)
        self.bn1 = nn.BatchNorm1d(self.fc1_size)
        #self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        #self.bn2 = nn.BatchNorm1d(self.fc2_size)

        self.fcs = nn.ModuleList()#[nn.Linear(self.fc1.out_features, self.fc1_size)])
        self.bns = nn.ModuleList()#[nn.BatchNorm1d(self.fc1_size)])
        prev_size = self.fc1_size
        prev_res_size = 0
        for _ in range(self.num_capsules):
            fc = nn.Linear(prev_size + prev_res_size, self.fc1_size)
            bn = nn.BatchNorm1d(self.fc1_size)
            self.fcs.append(fc)
            self.bns.append(bn)
            prev_res_size = prev_size
            prev_size = self.fc1_size

        self.fc_final = nn.Linear(prev_size, self.output_size)
        self.activation = nn.ELU()

        self.last_capsule_weights = np.zeros([self.num_capsules])

    def get_layer_usage(self, matrix: Variable):
        v1 = matrix.data[:, :self.fc1_size]
        v2 = matrix.data[:, self.fc1_size:]
        s1 = torch.mean(v1.abs())
        s2 = torch.mean(v2.abs())
        return s1 / (s1 + s2)

    def _forward(self, x):
        x = self.bn0(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)

        prev = [x]
        for fc, bn in zip(self.fcs, self.bns):
            c = torch.cat(prev[-2:], dim=-1)
            h = fc(c)
            #b = bn(h)
            a = self.activation(h)
            prev.append(a)

        out = self.fc_final(prev[-1])

        if x.shape[0] == 1:
            self.last_capsule_weights = [self.get_layer_usage(fc.weight)
                                         if fc.in_features > self.fc1_size
                                         else 1.0
                                         for fc in self.fcs]
            self.cou += 1

        return out

    def _forward_with_symmetry(self, x):
        direction = torch.sign(x[:, 0])
        x = direction.view(-1, 1) * x

        y = self._forward(x)

        swap_index = torch.cat([(direction < 0).view(-1, 1), (direction > 0).view(-1, 1)], 1).type(LongTensor)
        y = y.gather(1, swap_index)

        return y

    def forward(self, x):
        if self.use_symmetry:
            return self._forward_with_symmetry(x)
        else:
            return self._forward(x)


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
        self.start_lr = 1e-3
        self.stop_lr = 1e-9  # Used only with adaptive learning rate
        self.model = DQN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.start_lr, weight_decay=1e-5)
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
        loss = 0
        # if int(self.model.cou / 200) % 2:
        #     loss += 0.1 * torch.norm(self.model.c, 2)

        next_state_values = Variable(torch.zeros(state_var.data.size(0)).type(Tensor))
        next_state_values[non_final_mask] = self.model.eval()(non_final_next_states).max(1)[0]
        expected_action_values = (self.gamma * next_state_values) + reward_var

        # Huber loss
        loss += F.smooth_l1_loss(action_values, expected_action_values)
        #loss += torch.norm(self.model.c, 2, keepdim=True)
        #loss += torch.sum(self.model.c)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # if not int(self.model.cou / 200) % 2:
        #     self.model.fc1.weight.grad *= 0.0
        #     self.model.fc2.weight.grad *= 0.0
        #     self.model.fc3.weight.grad *= 0.0
        # else:
        #     self.model.fc_choice.weight.grad *= 0.0
        #     self.model.fc2.weight.grad *= 0.1
        #     self.model.fc3.weight.grad *= 0.1
        self.model.fc1.weight.grad *= 0.2

        self.optimizer.step()

    def adjust_learning_rate(self, ratio: float):
        self.optimizer.param_groups[0]['lr'] = self.stop_lr + (self.start_lr - self.stop_lr) * np.exp(-ratio)


def run():
    env = gym.make('CartPole-v0').unwrapped
    plt.ion()

    batch_size = 100
    actor = Actor()

    if use_cuda:
        actor.model.cuda()

    memory = ReplayMemory(10000)
    last_100_durations = deque(maxlen=100)

    plt.ion()
    fig = plt.figure()
    plot_buffer = deque(maxlen=30)
    plot_buffer.extend(np.ones([DQN.num_capsules]) for _ in range(30))
    capsule_heatmap = plt.imshow(np.array(plot_buffer), cmap='hot', interpolation='nearest')
    ax = plt.gca()
    ax.set_xlabel([0, 122])
    plt.clim(0, 1)
    fig.canvas.draw()
    plt.pause(0.1)

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

            if len(memory) < 2:
                last_100_durations.append(t)
                break

            batch = memory.sample(min(batch_size, len(memory)))
            actor.optimization_step(batch)

            plot_buffer.append(actor.model.last_capsule_weights)
            capsule_heatmap.set_array(np.array(plot_buffer))
            fig.canvas.draw()

            if done:
                last_100_durations.append(t)
                mean_duration = sum(last_100_durations) / len(last_100_durations)

                # Uncomment following line to turn the adaptive learning rate on
                actor.adjust_learning_rate(mean_duration / 100)
                print("Episode: {:4} | Duration: {:5} | Mean duration: {:8.3f} | Epsilon threshold: {:.6}".format(
                    i_episode, t, mean_duration, actor.decaying_binary_random.eps_threshold(actor.response_counter)))
                break

    print('Complete')
    env.close()
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run()
