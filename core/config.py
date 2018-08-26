import yaml
import logging
from core.utils import AttrDict

__C = AttrDict()
cfg = __C
## user can get config by from core.config import cfg
__C.checkpoint_cfg = AttrDict()
__C.checkpoint_cfg.model_name = 'baseline_dqn'
__C.checkpoint_cfg.model_file = 'exp/dqn_bs.pth'
__C.checkpoint_cfg.save_best = True

__C.running_settings = AttrDict()
__C.running_settings.use_cuda = True

__C.optim_params = AttrDict()
__C.optim_params.criteria_type = 'MSE'
__C.optim_params.optim_type = 'SGD'

__C.hyper_params = AttrDict()

__C.hyper_params.steps = 1000
__C.hyper_params.eval_steps = 1000

def yaml2cfg(yaml_file):
    with open(yaml_file, 'r') as f:
        cfg = yaml.load(f)

    print(cfg['checkpoint_cfg']['model_name'])
    return cfg
def test_load_cfg():
    import sys
    yaml_file = sys.argv[1]
    yaml2cfg(yaml_file)

if __name__ == '__main__':
    test_load_cfg()

