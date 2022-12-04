# han 2
import numpy as np
import torch
import math
import random
pi = torch.tensor(math.pi, dtype=torch.float)
class Han_Pendulum2():
    isTest = False

    def __init__(self, seed = 0):
        np.random.seed(seed)
        torch.manual_seed(seed) # set random seed
        random.seed(seed)

    def step(self, u):
        """go next step

        Args:
            u (torch.tensor([x])): torque

        Returns:
            _type_: state, reward, done, info
        """
        if self.isTest:
            print("u:", u)
        max_speed = torch.tensor(8.0, dtype=torch.float)#, requires_grad=True)
        max_torque = torch.tensor(2.0, dtype=torch.float)#, requires_grad=True)
        dt = torch.tensor(0.05, dtype=torch.float)#, requires_grad=True)
        g = torch.tensor(10.0, dtype=torch.float)#, requires_grad=True)
        m = torch.tensor(1.0, dtype=torch.float)#, requires_grad=True)
        l = torch.tensor(1.0, dtype=torch.float)#, requires_grad=True)
        
        ''' limit the range of u '''
        u = torch.clamp(u, -max_torque, max_torque)     
        u = torch.squeeze(u,0)
        u = torch.squeeze(u,0)
        # print(u)

        self.last_u = u  # for rendering
        ''' calculate cost before new state '''
        self.costs = angle_normalize(self.state[0].detach()) ** 2.0 + 0.1 * self.state[1].detach() ** 2.0 + 0.001 * (u ** 2.0)    # original, calculate reward before new state, (caused by action)
        self.newthdot = self.state[1].detach() + (3.0 * g / (2.0 * l) * torch.sin(self.state[0].detach()) + 3.0 / (m * l ** 2.0) * u) * dt
        self.newthdot_clamped = torch.clamp(self.newthdot, -max_speed, max_speed)       # limit the range of newthdot
        self.newth = self.state[0].detach() + self.newthdot_clamped * dt

        self.state = torch.stack([self.newth, self.newthdot_clamped])


        ''' calculate cost after new state '''
        return self._get_obs(), -self.costs, False, {} 

    def reset(self):
        # state[0]: theta
        # state[1]: theta_dot
        # The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.
        theta = torch.Tensor(1,1).uniform_(-math.pi, math.pi)
        theta_dot = torch.Tensor(1,1).uniform_(-1, 1)
        self.state = torch.stack([theta, theta_dot])
        self.state = torch.squeeze(self.state, 1)
        self.state = torch.squeeze(self.state, 1)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        result = torch.stack([torch.cos(self.state[0]), torch.sin(self.state[0]), self.state[1]])
        result = torch.unsqueeze(result,0)
        return result

    def set(self, state:torch.Tensor):
        """set a certain state to test

        Args:
            state (torch.Tensor): certain start state

        Returns:
            _type_: destached state
        """
        self.state = state
        return self._get_obs().detach()

def angle_normalize(x):
    return ((x + pi) % (2 * pi)) - pi
