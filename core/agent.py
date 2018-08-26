import torch
import logging
from core.config import yaml2cfg
from core.replay_memory import Transition
import sys
sys.path.append('/home/bingzhe/tetrisRL')
from engine import TetrisEngine
class base_agent(object):
    def __init__(self, args):
        self.agent_type = args['agent_type']
        ##checkpoint part 
        self.model_name = args['checkpoint_cfg']['model_name']
        self.model_file = args['checkpoint_cfg']['model_file']
        self.save_best = args['checkpoint_cfg']['save_best']
        if self.save_best:
            self.best_step = None
            self.best_reward = None

        ## running settings
        self.use_cuda = args['running_settings']['use_cuda']
        self.logger = args['running_settings']['logger']
        
        ## optimization part
        self.criteria_type = args['optim_params']['criteria_type']
        self.optimizer_type = args['optim_params']['optimizer_type']
        self.lr = args['optim_params']['init_lr']
        ## hyper parameters
        self.steps = args['hyper_params']['steps']
        self.early_stop = args['hyper_params']['early_stop']
        self.gamma = args['hyper_params']['gamma']
        self.eval_freq = args['hyper_params']['eval_freq']
        self.test_nepisodes = args['hyper_params']['test_nepisodes']
        self.eps = args['hyper_params']['eps']
        if self.agent_type == 'dqn':
            self.batch_size = args['dqn_params']['batch_size']
            self.capacity = args['dqn_params']['capacity']
        self.create_logger()
    def _reset_experience(self):
        self.transition = Transition(state0 = None,
                                     action = None,
                                     reward = None,
                                     state1 = None,
                                     terminal1 = False)
    def _load_model(self, model_file):
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
    def _save_model(self, step, curr_reward):
        if self.save_best:
            if self.best_step is None:
                self.best_step = step
                self.best_reward = curr_reward
            if curr_reward >= self.best_reward:
                self.best_Step = step
                self.best_reward = curr_reward
                torch.save(self.model.state_dict(), self.model_name)
        else:
            torch.save(self.model.state_dict(), self.model_name)
    def create_logger(self):
        if self.logger == "warnning":
            self.logger = logging.warning
    def creat_optimizer(self):
        if self.optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr)
    def _forward(self, observation):
        raise NotImplementedError("not implemented")
    def _backward(self, reward, terminal):
        pass
    def _eval_model(self):
        pass
    def fit_model(self):
        pass
    def test_model(self):
        pass
def test_base_agent():
    import sys 
    yaml_file = sys.argv[1]
    args = yaml2cfg(yaml_file)
    agent = base_agent(args)
    print(type(agent.batch_size))
    env = TetrisEngine(width, height)
    obs = env.clear()
    while True:
        action = 0
if __name__ == '__main__':
    test_base_agent()