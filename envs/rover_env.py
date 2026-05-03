import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math


class RoverEnv(gym.Env):

    def __init__(self, render_mode=None):
        super().__init__()

        # map dimensions
        self.L = 25.0 # length of the map L x L
        
        # dimension of rover
        self.W = 2.0 # width of the rover W x W

        # target region dimensions
        self.T = self.W # width of the target region. W x W

        # motion parameters
        self.v = 0.25 # forward velocity of the rover (m/step)
        
        # state of the rover
        self.x = 0.0 # x-coordinate of the rover's center
        self.y = 0.0 # y-coordinate of the rover's center
        self.theta = 0.0 # orientation of the rover (radians)
        self.goal = None # target region center (x_T, y_T)


        self.action_space = gym.spaces.Discrete(4)  # 4 possible actions: up, down, left, right


        # rendering parameters
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.scale = 30  # scale for rendering (pixels per meter)
        

    def reset(self):
        # Reset the state of the environment to an initial state
        # Return the initial observation
        pass

    def step(self, action):
        # Execute one time step within the environment
        # Return observation, reward, done, info
        pass

    def render(self):
        # Render the environment to the screen (optional)
        pass