import torch
import torch.nn as nn
import numpy as np

# I found this example CNN class online
class CNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.calc_conv_output(input_shape), 256),
            nn.ReLU(),
        )
        self.action = nn.Linear(256, num_actions)
        self.value = nn.Linear(256, 1)

    def calc_conv_output(self, shape):
        dummy_input = torch.zeros(1, *shape)
        dummy_output = self.conv_layers(dummy_input)
        return int(np.prod(dummy_output.size()))

    def forward(self, x):
        conv_out = self.conv_layers(x).view(x.size()[0], -1)
        fc_out = self.fc_layer(conv_out)
        return self.action(fc_out), self.value(fc_out)
