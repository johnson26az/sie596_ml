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

        # motion parameters. motion limits for PPO
        self.v_max = 0.5 # forward velocity of the rover (m/step)
        self.omega_max = 2.0 # rotation per step (radians/step)
        
        # state of the rover
        self.x = None # x-coordinate of the rover's center
        self.y = None # y-coordinate of the rover's center
        self.theta = None # orientation of the rover (radians)
        self.goal = None # target region center (x_T, y_T)

        # action space [v, omega]: forward velocity and rotation
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # observation space: [x, y, theta, x_T, y_T, d_edge, d_unit_x, d_unit_y]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, -np.pi, 0.0, 0.0, 0.0, -1, -1], dtype=np.float32),
            high=np.array([self.L, self.L, np.pi, self.L, self.L, self.L, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # time step limit
        self.dt = 0.1 # time step duration (seconds)

        # initialization contraints
        self.min_target_dist = self.W/2.0 # minimum distance between the target region and any boulder
        self.min_boulder_dist = 1.5 * (2 * self.R + self.W) # minimum distance between any two boulders to avoid overlap

        # rendering parameters
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.scale = 30  # scale for rendering (pixels per meter)

    '''
    Helper functions
    '''
    def _dist(self, p1, p2):
        return np.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    def _sample_point(self):
        return self.np_random.uniform(self.r_boundary, self.L - self.r_boundary, size=2)


    '''
    Reset the state of the environment to an initial state
    Return the initial observation
    '''
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # rover position and orientation
        self.x = self._sample_point()
        self.y = self._sample_point()
        self.theta = self.np_random.uniform(-np.pi, np.pi)

        # target position (ensure it's not too close to the boundary to allow the rover to fit)
        while True:
            self.goal = self._sample_point()
            if self._dist(self.goal, [self.x, self.y]) > self.min_target_dist:
                break

        # boulder's placement with constraints
        self.boulders = []
        for i in range(self.N):
            while True:
                boulder_pos = self._sample_point()

                # check distance from target region
                if self._dist(boulder_pos, self.goal) < self.min_target_dist:
                    continue

                # check distance from rover's initial position
                if self._dist(boulder_pos, [self.x, self.y]) < self.min_target_dist:
                    continue

                # check distance from other boulders
                if any(self._dist(boulder_pos, existing) < self.min_boulder_dist for existing in self.boulders):
                    continue

                self.boulders.append(boulder_pos)
                break

        # initial boulder features (relative to the rover's center)
        self.d_edge, self.d_unit = self._compute_boulder_features()

        obs = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return obs, {}


    '''
    Step with continuous action input [v, omega]: forward velocity and rotation
    '''
    def step(self, action):
        # neural network output to normalized wheel commands
        u_R = float(np.clip(action[0], -1.0, 1.0))
        u_L = float(np.clip(action[1], -1.0, 1.0))

        # convert normalized commands to wheel angular velocities
        omega_R = self.omega_max * u_R
        omega_L = self.omega_max * u_L

        # convert wheel angular velocities to rover's linear and angular velocity
        # self.wr = 0.25 wheel radius, self.rW = 2.0 rover width
        v = self.wr * (omega_R + omega_L) / 2.0
        omega = self.wr * (omega_R - omega_L) / self.rW

        # unicycle model for rover motion
        self.x += v * math.cos(self.theta) * self.dt
        self.y += v * math.sin(self.theta) * self.dt
        self.theta += omega * self.dt
        # normalize heading angle to [-pi, pi]
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        # base reward is negative distance to the target
        reward = -0.01
        terminated = False
        truncated = False

        # target check
        if self._in_target_region():
            reward = 200.0
            terminated = True

        # collision check
        if self._obstabcle_collision() or self._boundary_collision():
            reward = -200.0
            terminated = True

        obs = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, {}


    '''
    Check for target region
    '''
    def _in_target_region(self):
        half_T = self.T / 2.0
        return (
            self.goal[0] - half_T <= self.x <= self.goal[0] + half_T and
            self.goal[1] - half_T <= self.y <= self.goal[1] + half_T
        )


    '''
    Clean up resources when the environment is closed
    '''
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
        return super().close()


    '''
    Function to get the current observation of the environment
    '''
    def _get_obs(self):
        d_edge, d_unit = self._compute_boulder_features()

        return np.array(
            [   self.x,
                self.y,
                self.theta,
                self.goal[0],
                self.goal[1],
                d_edge,
                d_unit[0],
                d_unit[1]
            ],
            dtype=np.float32)


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
    Check for collision with the boundary of the map
    '''
    def _boundary_collision(self):
        return (
            self.x < self.r_boundary or
            self.x > self.L - self.r_boundary or
            self.y < self.r_boundary or
            self.y > self.L - self.r_boundary
        )
    

    '''
    Calculating boulder features: distance to the nearest boulder edge and unit vector pointing to the nearest boulder (relative to the rover's center)
    '''
    def _compute_boulder_features(self):
        nearest_dist = float('inf')
        nearest_vec = np.array([0.0, 0.0], dtype=np.float32)

        for bx, by in self.boulders:
            vec = np.array([bx - self.x, by - self.y], dtype=np.float32)
            dist_center = np.linalg.norm(vec)

            if dist_center < nearest_dist:
                nearest_dist = dist_center
                nearest_vec = vec

        # distance to boulder edge
        d_edge = nearest_dist - self.R

        # unit vector to boulder
        if nearest_dist > 0:
            d_unit = nearest_vec / nearest_dist
        else:
            d_unit = np.array([0.0, 0.0], dtype=np.float32)

        return float(d_edge), d_unit.astype(np.float32)


    '''
     Render the environment to the screen
    '''
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

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

        # draw boulder features
        boulder_x = int((self.x + self.d_unit[0] * self.d_edge) * self.scale)
        boulder_y = int((self.y + self.d_unit[1] * self.d_edge) * self.scale)
        pygame.draw.circle(self.screen, (255, 0, 0), (boulder_x, boulder_y), 5)  # red boulder feature point

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])