"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import torch
import torch.nn as nn
from utils import img_wrap, set_init, push_and_pull_atari, record, pre_proc
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import os
os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 600
GAMMA = 0.9
MAX_EP = 12000

env = gym.make('BreakoutDeterministic-v4')
N_S = env.observation_space.shape[0]
N_A = env.action_space.n

# Box(210, 160, 3) Discrete(4)

class Net(nn.Module):
    def __init__(self, a_dim):
        super(Net, self).__init__()
        self.a_dim = a_dim

        self.conv1 = nn.Conv2d(1 , 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, self.a_dim)
        self.fc6 = nn.Sequential(nn.Linear(512, 128),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(128, 1))

        #set_init([self.conv1, self.conv2, self.conv3, self.fc4, self.fc5, self.fc6])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        logits = self.fc5(x)
        values = self.fc6(x)

        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data

        #print('nan values:', torch.sum(torch.isnan(prob)).item())
        if torch.sum(torch.isnan(prob)).item() > 0:
            print("! nan ")
            print(prob)
            exit()

        m = self.distribution(prob)
        return m.sample().numpy()[0]
        #return torch.argmax(prob)

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_A)           # local network
        self.env = gym.make('BreakoutDeterministic-v4').unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            s = pre_proc(s)
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                if self.name == 'w00':
                    self.env.render()
                a = self.lnet.choose_action(img_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                s_ = pre_proc(s_)

                if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull_atari(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        print(" + ",self.name," done !")
                        break
                s = s_
                total_step += 1

        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = Net(N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
