from collections import deque
from torchinfo import summary
import numpy as np
import random
import torch
import torch.nn as nn


class DQN(nn.Module): 
    
    def __init__(self, input, max_exp, action_space, device):
        
        self.input_space = input
        self.device = device
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=4, device=device)
        self.relu_conv1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, device=device)
        self.relu_conv2 = nn.ReLU()
        
        self.fc1 = nn.Linear(13824, 256, device=device)
        self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(1850, 1100)
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(1100, 500)
        # self.relu3 = nn.ReLU()
        # self.fc4 = nn.Linear(500, 256)
        # self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(256, action_space, device=device)
        
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        self.experiences = deque(maxlen=max_exp)
        
    def __str__(self):
        return f"{summary(self, self.input_space)}"
    
    def convert(self, x):
        # turn a numpy arrya into float32 and add a dim to it
        return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
    
    def forward(self, x):
        
        out = self.conv1(x) 
        out = self.relu_conv1(out)
        out = self.conv2(out)
        out = self.relu_conv2(out)
        
        out = self.fc1(torch.flatten(out))
        out = self.relu1(out)
        
        # out = self.fc2(out)
        # out = self.relu2(out)
        
        # out = self.fc3(out)
        # out = self.relu3(out)
        
        # out = self.fc4(out)
        # out = self.relu4(out)
        
        return self.fc5(out)
    
    def find_targets(self, batch_size, gamma):
        
        minibatch = random.choices(self.experiences, k=batch_size)
        
        targets = torch.zeros(len(minibatch))
        current = torch.zeros(len(minibatch))
        rewards = torch.zeros(len(minibatch))
        
        for i in range(len(minibatch)):
            
            is_terminal = minibatch[i].terminal
            r = minibatch[i].reward
            s = minibatch[i].state
            next_s = minibatch[i].next_state
            a = minibatch[i].action
            
            if is_terminal:
                yj = r
            else:
                with torch.no_grad():
                    yj = r + gamma * torch.max((self.forward(self.convert(next_s).to(self.device))))
                
            # Perform a gradient descent step on (yj − Q(φj , aj ; θ))^2 according to equation 3
            Q_sa = self.forward(self.convert(s).to(self.device))
            
            targets[i] = yj
            current[i] = Q_sa[a]
            rewards[i] = r
        
        return targets, current, rewards

                