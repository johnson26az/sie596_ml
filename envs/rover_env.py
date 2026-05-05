import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import pygame


class RoverEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # map dimensions
        self.L = 25.0 # length of the map L x L

        # target region dimensions
        self.T = self.W # width of the target region. W x W
        
        # boulder's dimensions
        self.N = 6 # number of boulders
        self.R = 1.0 # radius of each boulder (for collision detection)
        self.boulders = None # list of boulders' positions [(x1, y1), (x2, y2), ..., (xN, yN)]

        # parameters of the rover
        self.W = 2.0 # width of the rover W x W
        self.d_safe = self.R + (np.sqrt(2)/2)*self.W # minimum safe distance from the center of a boulder to the center of the rover
        self.r_boundary = self.W / np.sqrt(2) # minimum distance from the center of the rover to the boundary of the map to avoid collision with the boulders

        # motion parameters
        self.v = 0.25 # forward velocity of the rover (m/step)
        self.omega = np.deg2rad(10) # rotation per step (radians/step)
        
        # state of the rover
        self.x = 0.0 # x-coordinate of the rover's center
        self.y = 0.0 # y-coordinate of the rover's center
        self.theta = 0.0 # orientation of the rover (radians)
        self.goal = None # target region center (x_T, y_T)

        self.lidar_reading = np.array([0.0,0.0], dtype=np.float32) # point detected by the lidar (relative to the rover's center)

        self.action_space = gym.spaces.Discrete(4)  # 4 possible actions: up, down, left, right


        # rendering parameters
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.scale = 30  # scale for rendering (pixels per meter)
    
    '''
    Reset the state of the environment to an initial state
    Return the initial observation
    '''
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # rover pose
        self.x = self.np_random.uniform(self.r_boundary, self.L - self.r_boundary)
        self.y = self.np_random.uniform(self.r_boundary, self.L - self.r_boundary)
        self.theta = self.np_random.uniform(-np.pi, np.pi)

        # target center
        self.goal = self.np_random.uniform(self.r_boundary, self.L - self.r_boundary, size=2)

        # boulder's positions
        self.boulders = self.np_random.uniform(
            self.R + self.d_safe,
            self.L - self.R - self.d_safe,
            size=(self.N, 2)
        )

        # initial lidar reading (relative to the rover's center)
        self.lidar_reading = self._compute_lidar_reading()

        obs = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return obs, {}

    '''
    Execute one time step within the environment
    '''
    def step(self, action):
        # Execute one time step within the environment
        # Return observation, reward, done, info
        pass

    '''
     Render the environment to the screen
    '''
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def close(self):
        return super().close()


    '''
    Function to get the current observation of the environment
    '''
    def _get_obs(self):
        return np.array([self.x, self.y, self.theta, self.goal[0], self.goal[1]], dtype=np.float32)

    '''
    Functions for collision detection
    '''
    def _obstabcle_collision(self):
        for bx, by in self.boulders:
            dist = np.hypot(self.x - bx, self.y - by)
            if dist < self.d_safe:
                return True
        return False
    
    '''
    Function to compute the lidar reading (relative to the rover's center)
    '''
    def _compute_lidar_reading(self):
        nearest_point = np.array([self.x, self.y], dtype=np.float32)
        nearest_dist = float('inf')

        for bx, by in self.boulders:
            dx = self.x - bx
            dy = self.y - by
            dist = math.hypot(dx, dy)

            if dist == 0:
                continue

            # closest point on boulder surface
            px = bx + (self.R * dx / dist)
            py = by + (self.R * dy / dist)

            if dist < nearest_dist:
                nearest_dist = dist
                nearest_point = np.array([px, py], dtype=np.float32)

        return nearest_point

    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((int(self.L * self.scale), int(self.L * self.scale)))
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))  # white background

        # draw target region
        target_rect = pygame.Rect(
            int((self.goal[0] - self.T/2) * self.scale),
            int((self.goal[1] - self.T/2) * self.scale),
            int(self.T * self.scale),
            int(self.T * self.scale)
        )
        pygame.draw.rect(self.screen, (0, 255, 0), target_rect)  # green target region

        # draw boulders
        for bx, by in self.boulders:
            pygame.draw.circle(
                self.screen,
                (128, 128, 128),  # gray boulders
                (int(bx * self.scale), int(by * self.scale)),
                int(self.R * self.scale)
            )

        # draw rover
        rover_rect = pygame.Rect(
            int((self.x - self.W/2) * self.scale),
            int((self.y - self.W/2) * self.scale),
            int(self.W * self.scale),
            int(self.W * self.scale)
        )
        pygame.draw.rect(self.screen, (0, 0, 255), rover_rect)  # blue rover

        # draw lidar reading
        lidar_x = int((self.x + self.lidar_reading[0]) * self.scale)
        lidar_y = int((self.y + self.lidar_reading[1]) * self.scale)
        pygame.draw.circle(self.screen, (255, 0, 0), (lidar_x, lidar_y), 5)  # red lidar point

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])