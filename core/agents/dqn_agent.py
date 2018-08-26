import sys
sys.path.append('/home/bingzhe/tetris-rl-collection')
sys.path.append('/home/bingzhe/tetrisRL')
from engine import TetrisEngine
from core.agent import base_agent
from core.replay_memory import ReplayMemory
from core.replay_memory import Transition
import torch.nn as nn
import torch.optim as optim
from core.networks import dqn_network
from core.config import yaml2cfg
from itertools import count
import random
import torch.nn.functional as F
import torch
class dqn_agent(base_agent):
    def __init__(self, args):
        super(dqn_agent, self).__init__(args)
        print(self.logger)
        self.env = TetrisEngine(10, 20)
        self.memory = ReplayMemory(self.capacity)
        self.model = dqn_network()
        if self.use_cuda:
            self.model.cuda()
        if self.criteria_type == 'MSE':
            self.loss = nn.MSELoss()
        if self.optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr)
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_cuda else torch.ByteTensor
        self.steps_done = 0
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps
        self.steps_done += 1
        if sample > eps_threshold:
            return self.model(self.FloatTensor(state)).data.max(1)[1].view(1, 1)
        else:
            return self.FloatTensor([[random.randrange(7)]])
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = self.ByteTensor(tuple(map(lambda s : s is not None,
                                                    batch.state1)))
        with torch.no_grad():
            non_final_next_states = torch.cat([s for s in batch.state1 if s is not None])
        state_batch = torch.cat(batch.state0)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        ## compute Q(s_t, a) 
        state_action_values = self.model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size).type(self.FloatTensor)
        next_state_values[non_final_mask] = self.model(non_final_next_states).detach().max(1)[0]
        next_state_values.volatile = False
        expected_state_action_values = (next_state_values*self.gamma) + reward_batch
        #print(state_action_values.size())
        state_action_values = state_action_values.view(-1)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss
        if len(loss.data) > 0: return loss.data[0]
        else: return loss

    def fit_model(self):
        start_epoch = 0
        best_score = 0
        f = open('log.out', 'w+')
        self.step_done = 0
        for i_episode in count(start_epoch):
            state = self.FloatTensor(self.env.clear()[None, None, :, :])
            score = 0
            for t in count():
                self.step_done += 1
                action = self.select_action(state).type(self.LongTensor)
                last_state = state
                state, reward, done = self.env.step(int(action[0,0].cpu().numpy()))
                state = self.FloatTensor(state[None, None, :, :])
                score += int(reward)
                reward = self.FloatTensor([float(reward)])
                self.memory.push(last_state, action, state, reward, done)
                if done:
                    if i_episode % 10 == 0:
                        log = 'epoch {0} score {1}'.format(i_episode, score)
                        self.logger(log)
                        f.write(log + '\n')
                        loss = self.optimize_model()
                        if loss:
                            self.logger('loss: {:.0f}'.format(loss))
                    if i_episode % 100 == 0:
                        self._save_model(self.step_done, score,)
                    break
        f.close()
        print('Complete')
    def test_model(self):
        self.model.load_state_dict(torch.load(self.model_name))
        obs = self.env.clear()[None, None, :, :]
        game_len = 0
        score = 0
        while True:
            game_len += 1
            action = self.select_action(obs)
            action = action.view(-1).cpu().numpy()[0]
            obs, reward, done = self.env.step(action)
            score += reward
            obs = obs[None, None, :, :]
            if done :
                break
        print(score)
        print(game_len)

def test_dqn_agent():
    import sys
    yaml_file = sys.argv[1]
    cfg = yaml2cfg(yaml_file)
    dqn_ = dqn_agent(cfg)
    print(dqn_.model)
    env = TetrisEngine(10, 20)
    obs = env.clear()[None, None, :, :]
    dqn_.model.cuda()
    while True:
        action = dqn_.select_action(obs)
        action = action.view(-1).cpu().numpy()[0]
        obs, reward, done = env.step(action)
        obs = obs[None, None, :,:]
        print(obs.shape)
        if done:
            break 
def test_dqn_training():
    import sys
    yaml_file = sys.argv[1]
    cfg = yaml2cfg(yaml_file)
    dqn_ = dqn_agent(cfg)
    dqn_.fit_model()
def test_dqn_eval():
    import sys
    yaml_file = sys.argv[1]
    cfg = yaml2cfg(yaml_file)
    dqn_ = dqn_agent(cfg)
    dqn_.test_model()
if __name__ == '__main__':
    #test_dqn_training()
    test_dqn_eval()