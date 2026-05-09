import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import pygame
import scipy.integrate as integrate

from envs import max_time


class RoverEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, l=None, W=None, N=None, R=None, r=None, rW=None, omega_max=None, delta_t=None, max_time_steps=None, reward_params=None):
        super().__init__()

        # Use provided values or defaults - values should come from config
        l = l if l is not None else 25.0
        W = W if W is not None else 2.0
        N = N if N is not None else 6
        R = R if R is not None else 1.0
        r = r if r is not None else 0.25
        rW = rW if rW is not None else 2.0
        omega_max = omega_max if omega_max is not None else 2.0
        delta_t = delta_t if delta_t is not None else 0.1
        reward_params = reward_params or {}

        # map dimensions
        self.L = l # length of the map L x L

        # parameters of the rover
        self.W = W # width of the rover W x W
        self.T = self.W # width of the target region. W x W
        self.r_boundary = self.W / np.sqrt(2) # minimum distance from the center of the rover to the boundary of the map to avoid collision with the boulders

        # boulder's dimensions
        self.N = N # number of boulders
        self.R = R # radius of each boulder (for collision detection)
        self.boulders = None # list of boulders' positions [(x1, y1), (x2, y2), ..., (xN, yN)]
        self.d_safe = self.R + (np.sqrt(2)/2)*self.W # minimum safe distance from the center of a boulder to the center of the rover

        # motion parameters. motion limits for PPO
        self.v_max = 0.5 # forward velocity of the rover (m/step)
        self.omega_max = omega_max # rotation per step (radians/step)
        self.wr = r
        self.rW = rW
        
        # state of the rover
        self.x = None # x-coordinate of the rover's center
        self.y = None # y-coordinate of the rover's center
        self.theta = None # orientation of the rover (radians)
        self.goal = None # target region center (x_T, y_T)
        self.prev_v = 0.0 # previous forward velocity for reward shaping
        self.prev_theta = 0.0 # previous heading for reward shaping
        self.prev_u_R = 0.0 # previous right wheel command for action smoothness
        self.prev_u_L = 0.0 # previous left wheel command for action smoothness

        # action space [u_R, u_L]: normalized wheel angular commands (right, left) in [-1,1]
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
        self.dt = delta_t # time step duration (seconds)

        # initialization contraints
        self.min_target_dist = 1.5 * (2 * self.R + self.W) # minimum distance between the target region and any boulder
        self.min_boulder_dist = 1.5 * (2 * self.R + self.W) # minimum distance between any two boulders to avoid overlap
        self.max_time = max_time_steps if max_time_steps is not None else max_time

        # reward shaping weights; tuned to favor earlier goal arrival
        self.reward_progress_scale = float(reward_params.get('progress_scale', 4.5))
        self.reward_time_penalty = float(reward_params.get('time_penalty', 0.08))
        self.reward_speed_bonus = float(reward_params.get('speed_bonus', 0.06))
        self.reward_slow_speed_threshold = float(reward_params.get('slow_speed_threshold', 0.08))
        self.reward_slow_speed_penalty = float(reward_params.get('slow_speed_penalty', 0.25))
        self.reward_boulder_penalty_scale = float(reward_params.get('boulder_penalty_scale', 2.0))
        self.reward_alignment_penalty = float(reward_params.get('alignment_penalty', 0.05))
        self.reward_reverse_penalty = float(reward_params.get('reverse_penalty', 0.5))
        self.reward_angular_penalty = float(reward_params.get('angular_penalty', 0.10))
        self.reward_heading_change_penalty = float(reward_params.get('heading_change_penalty', 0.03))
        self.reward_action_smoothness_penalty = float(reward_params.get('action_smoothness_penalty', 0.08))
        self.reward_alignment_bonus = float(reward_params.get('alignment_bonus', 0.10))
        self.reward_goal_bonus = float(reward_params.get('goal_bonus', 200.0))
        self.reward_fast_finish_bonus = float(reward_params.get('fast_finish_bonus', 120.0))

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
        self.x = self.np_random.uniform(self.r_boundary, self.L - self.r_boundary)
        self.y = self.np_random.uniform(self.r_boundary, self.L - self.r_boundary)
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

        self.sim_time = 0.0
        self.prev_v = 0.0
        self.prev_theta = self.theta
        self.prev_u_R = 0.0
        self.prev_u_L = 0.0

        obs = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return obs, {}


    '''
    Step with continuous action input [v, omega]: forward velocity and rotation
    '''
    def step(self, action):
        # neural network output: normalized wheel commands (right, left)
        u_R = float(np.clip(action[0], -1.0, 1.0))
        u_L = float(np.clip(action[1], -1.0, 1.0))

        # convert normalized commands to wheel angular velocities
        omega_R = self.omega_max * u_R
        omega_L = self.omega_max * u_L

        # convert wheel angular velocities to rover's linear and angular velocity
        # self.wr = wheel radius, self.rW = rover width
        v = self.wr * (omega_R + omega_L) / 2.0
        omega = self.wr * (omega_R - omega_L) / self.rW

        # store current velocities for rendering
        self.v = v
        self.omega = omega

        # save old position for reward shaping
        self.old_x, self.old_y = self.x, self.y

        # integrate continuous-time dynamics over the time step duration
        state0 = [self.x, self.y, self.theta]
        sol = integrate.solve_ivp(
            fun=lambda t, state: self._dynamics(t, state, v, omega),
            t_span=[0, self.dt],
            y0=state0,
            method='RK45'
        )

        self.x, self.y, self.theta = sol.y[:, -1]
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi  # wrap angle to [-pi, pi]

        # base reward is negative distance to the target
        terminated = False
        truncated = False

        # update simulation time
        self.sim_time += self.dt



        # reward shaping: reward for getting closer to the target region compared to the previous step
        # computing the reward
        reward = 0.0

        # distance to goal shaping
        old_dist = np.linalg.norm([self.old_x - self.goal[0], self.old_y - self.goal[1]])
        new_dist = np.linalg.norm([self.x - self.goal[0], self.y - self.goal[1]])
        reward += self.reward_progress_scale * (old_dist - new_dist)  # reward for getting closer to the target

        # boulder proximity penalty (encourage the rover to stay away from boulders)
        self.d_edge, _ = self._compute_boulder_features()
        if self.d_edge < 2.0:  # if the rover is within 2 meters of a boulder edge
            reward -= (2.0 - self.d_edge) * self.reward_boulder_penalty_scale  # penalty increases as the rover gets closer to the boulder

        # control effort penalty (encourage energy-efficient solutions)
        reward -= 0.001 * (u_R**2 + u_L**2)

        # time penalty (encourage faster solutions)
        reward -= self.reward_time_penalty

        # forward speed bonus (encourage the rover to keep moving forward)
        speed = abs(v)
        reward += self.reward_speed_bonus * speed

        # penalty for being to slow (encourage the rover to maintain a minimum speed)
        if speed < self.reward_slow_speed_threshold:
            reward -= self.reward_slow_speed_penalty

        # direction-consistent bonus (encourage the rover to maintain a consistent heading towards the target)
        self.goal_vector = np.array([self.goal[0] - self.x, self.goal[1] - self.y])
        self.heading_vector = np.array([math.cos(self.theta), math.sin(self.theta)])
        alignment = np.dot(self.goal_vector, self.heading_vector) / (np.linalg.norm(self.goal_vector) + 1e-6)

        # penalize when facing away from the target and reward when facing towards the target
        if alignment < 0:
            reward -= self.reward_alignment_penalty  # small penalty for facing away from the target

        # penalize when reversing direction (encourage the rover to maintain a consistent heading towards the target)
        if np.sign(v) != np.sign(self.prev_v):
            reward -= self.reward_reverse_penalty  # increased penalty for reversing direction
        self.prev_v = v

        # penalized for large angular velocity (encourage smoother trajectories) - INCREASED
        reward -= self.reward_angular_penalty * abs(omega)

        # penalized for rapid heading changes (encourage smoother trajectories)
        heading_change = abs(self.theta - getattr(self, "prev_theta", self.theta))
        reward -= self.reward_heading_change_penalty * heading_change
        self.prev_theta = self.theta

        # penalize action smoothness (rapid changes in steering commands)
        action_smoothness = abs(u_R - self.prev_u_R) + abs(u_L - self.prev_u_L)
        reward -= self.reward_action_smoothness_penalty * action_smoothness
        self.prev_u_R = u_R
        self.prev_u_L = u_L

        # bonus for consistent forward movement (encourage the rover to maintain a consistent heading towards the target)\
        if v > 0 and alignment > 0.3:
            reward += self.reward_alignment_bonus * alignment  # bonus for facing and moving towards target (only if well-aligned)



        # termination conditions
        # target check
        if self._in_target_region():
            finish_fraction = max(0.0, 1.0 - (self.sim_time / max(self.max_time, 1e-6)))
            reward += self.reward_goal_bonus + (self.reward_fast_finish_bonus * finish_fraction)
            terminated = True

        # collision check
        elif self._obstabcle_collision() or self._boundary_collision():
            reward = -200.0
            terminated = True

        elif self.sim_time >= self.max_time:
            reward = -100.0
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
    Continious-time dynamics function for the rover given the current state and action
    '''
    def _dynamics(self, t, state, v, omega):
        x, y, theta = state
        dxdt = v * math.cos(theta)
        dydt = v * math.sin(theta)
        dthetadt = omega
        return np.array([dxdt, dydt, dthetadt], dtype=np.float32)


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

        # beige/tan background (sand/dirt map)
        self.screen.fill((230, 200, 150))

        # draw target region (blue square)
        target_rect = pygame.Rect(
            int((self.goal[0] - self.T/2) * self.scale),
            int((self.goal[1] - self.T/2) * self.scale),
            int(self.T * self.scale),
            int(self.T * self.scale)
        )
        pygame.draw.rect(self.screen, (0, 0, 200), target_rect)

        # draw boulders (brown) and danger areas (black outlines)
        for bx, by in self.boulders:
            center_px = (int(bx * self.scale), int(by * self.scale))
            radius_px = int(self.R * self.scale)
            # filled boulder: brown
            pygame.draw.circle(self.screen, (150, 75, 0), center_px, radius_px)
            # danger area: outline in black (radius = R + W)
            danger_radius_m = self.R + self.W
            danger_radius_px = int(danger_radius_m * self.scale)
            if danger_radius_px > radius_px:
                pygame.draw.circle(self.screen, (0, 0, 0), center_px, danger_radius_px, width=2)

        # draw rover as a gray square
        rover_size = int(self.W * self.scale)
        rover_surface = pygame.Surface((rover_size, rover_size), pygame.SRCALPHA)
        pygame.draw.rect(rover_surface, (200, 200, 200), rover_surface.get_rect())  # gray

        rotated_rover = pygame.transform.rotate(rover_surface, -math.degrees(self.theta))
        rover_rect = rotated_rover.get_rect(center=(int(self.x * self.scale), int(self.y * self.scale)))
        self.screen.blit(rotated_rover, rover_rect)

        # draw velocity arrow (green) from rover center in heading direction
        try:
            v = float(getattr(self, 'v', 0.0))
        except Exception:
            v = 0.0
        arrow_length_px = int(max(6, v * self.scale * 4))  # scale so small v still visible
        cx, cy = int(self.x * self.scale), int(self.y * self.scale)
        end_x = int(cx + arrow_length_px * math.cos(self.theta))
        end_y = int(cy + arrow_length_px * math.sin(self.theta))
        pygame.draw.line(self.screen, (0, 200, 0), (cx, cy), (end_x, end_y), width=3)
        # arrowhead
        ah_size = max(6, arrow_length_px // 4)
        left = (int(end_x - ah_size * math.cos(self.theta - math.pi / 6)), int(end_y - ah_size * math.sin(self.theta - math.pi / 6)))
        right = (int(end_x - ah_size * math.cos(self.theta + math.pi / 6)), int(end_y - ah_size * math.sin(self.theta + math.pi / 6)))
        pygame.draw.polygon(self.screen, (0, 200, 0), [(end_x, end_y), left, right])

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            # return RGB array if needed (optional)
            return pygame.surfarray.array3d(self.screen)