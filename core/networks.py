import torch.nn as nn
import sys
import torch
sys.path.append('/home/bingzhe/tetrisRL')
import engine

class base_network(nn.Module):
    """
    Base class for feature extracting
    """

    def __init__(self, hidden_layers = None):
        self.hidden_layers = hidden_layers
    
def conv_block(in_channels, out_channels, kernel_size, stride = 1):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )
class dqn_network(nn.Module):
    def __init__(self):
        super(dqn_network, self).__init__()
        self.conv1 = conv_block(1, 16, kernel_size = 3, stride = 1)
        self.conv2 = conv_block(16, 32, 4, 2)
        self.linear1 = nn.Linear(768, 256)
        self.head = nn.Linear(256, 7)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.linear1(x))
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x
def test_dqn_network():
    net = dqn_network()
    inputs = torch.zeros((1, 1, 10, 20))
    out = net(inputs)
    print(out.size())
if __name__ == '__main__':
    test_dqn_network()