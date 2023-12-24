from collections import deque
from torchinfo import summary
import numpy as np
import random
import torch
import torch.nn as nn


class DQN(nn.Module): 
    
    def __init__(self, input, max_exp, action_space):
        
        self.input_space = input
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=2, padding=4)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=8, stride=2, padding=0)
        
        self.fc1 = nn.Linear(3626, 2460)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2460, 1800)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1800, 700)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(700, 120)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(120, action_space)
        
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
        self.experiences = deque(maxlen=max_exp)
        
    def __str__(self):
        return f"{summary(self, self.input_space)}"
    
    def convert(self, x):
        # turn a numpy arrya into float32 and add a dim to it
        return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
    
    def forward(self, x):
        
        out = self.conv1(x) 
        out = self.pool(out)       
        out = self.conv2(out)
        
        out = self.fc1(torch.flatten(out))
        out = self.relu1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        
        out = self.fc3(out)
        out = self.relu3(out)
        
        out = self.fc4(out)
        out = self.relu4(out)
        
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
                    yj = r + gamma * torch.max((self.forward(self.convert(next_s))))
                
            # Perform a gradient descent step on (yj − Q(φj , aj ; θ))^2 according to equation 3
            Q_sa = self.forward(self.convert(s))
            
            targets[i] = yj
            current[i] = Q_sa[a]
            rewards[i] = r
        
        return targets, current, rewards

                