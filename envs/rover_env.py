import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import pygame
import scipy.integrate as integrate

from envs import max_time
from envs.a_star_planner import astar


class RoverEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, l=None, W=None, N=None, R=None, r=None, rW=None, omega_max=None, delta_t=None, max_time_steps=None):
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

        # rendering parameters
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.scale = 30  # scale for rendering (pixels per meter)
        # grid/world bounds for path planning
        self.x_min = 0.0
        self.x_max = self.L
        self.y_min = 0.0
        self.y_max = self.L
        self.grid_resolution = 0.5

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
        # seed RNG first
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

                # store boulder as (x,y, radius)
                self.boulders.append((boulder_pos[0], boulder_pos[1], self.R))
                break


        '''
        Path planning with A*: we will create a grid representation of the environment, marking the boulders as obstacles, and then run A* to find a path from the rover's initial position to the target region. The resulting path will be stored as a list of waypoints that the rover can follow during the episode.
        '''
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max

        nx = max(1, int((x_max - x_min) / self.grid_resolution))
        ny = max(1, int((y_max - y_min) / self.grid_resolution))
        grid = np.zeros((ny, nx), dtype=np.int32)

        # mark boulders on the grid
        for (bx, by, br) in self.boulders:
            for iy in range(ny):
                for ix in range(nx):
                    wx = x_min + (ix + 0.5) * self.grid_resolution
                    wy = y_min + (iy + 0.5) * self.grid_resolution
                    if np.hypot(wx - bx, wy - by) <= br + 0.3:
                        grid[iy, ix] = 1  # mark as obstacle
        self.grid = grid

        # compute start and goal indices for A*
        def world_to_grid(x, y):
            ix = int((x - x_min) / self.grid_resolution)
            iy = int((y - y_min) / self.grid_resolution)
            ix = np.clip(ix, 0, nx - 1)
            iy = np.clip(iy, 0, ny - 1)
            return ix, iy

        start_idx = world_to_grid(self.x, self.y)
        goal_idx = world_to_grid(self.goal[0], self.goal[1])

        # run A* to find a path from start to goal (astar expects (ix,iy) pairs)
        path_idx = astar(self.grid, start_idx, goal_idx)

        # convert path to world waypoints
        self.path = []
        for ix, iy in path_idx:
            wx = x_min + (ix + 0.5) * self.grid_resolution
            wy = y_min + (iy + 0.5) * self.grid_resolution
            self.path.append((wx, wy))

        self.current_waypoint_idx = 0

        # initial boulder features (relative to the rover's center)
        self.d_edge, self.d_unit = self._compute_boulder_features()

        self.sim_time = 0.0
        self.prev_v = 0.0
        self.prev_theta = self.theta
        # initialize previous position used in reward shaping
        self.old_x, self.old_y = self.x, self.y

        obs = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return obs, {}


    '''
    Step with continuous action input [v, omega]: forward velocity and rotation
    '''
    def step(self, action):

        '''
        waypoint guidance reward shaping: provide a reward based on the rover's progress towards the next waypoint in the A* path. This encourages the agent to follow the path while still allowing for some flexibility in navigation.
        '''
        # initialize reward accumulator early (avoids UnboundLocalError)
        reward = 0.0

        # ensure path has at least the goal as a waypoint
        if not getattr(self, 'path', None):
            self.path = [(self.goal[0], self.goal[1])]

        # safe previous position defaults
        old_x = getattr(self, 'old_x', self.x)
        old_y = getattr(self, 'old_y', self.y)

        wp_x, wp_y = self.path[self.current_waypoint_idx]

        old_wp_dist = np.linalg.norm([old_x - wp_x, old_y - wp_y])
        new_wp_dist = np.linalg.norm([self.x - wp_x, self.y - wp_y])

        # reward for getting closer to the next waypoint
        reward += 2.0 * (old_wp_dist - new_wp_dist)

        # advance waypoint if the rover is close enough to the current waypoint
        if new_wp_dist < 0.5 and self.current_waypoint_idx < len(self.path) - 1:
            self.current_waypoint_idx += 1
            wp_x, wp_y = self.path[self.current_waypoint_idx]

        # heading alignment reward: provide a reward based on how well the rover's heading aligns with the direction to the next waypoint. This encourages the agent to orient itself towards the path.
        wp_heading = np.arctan2(wp_y - self.y, wp_x - self.x)
        heading_error = abs(self.theta - wp_heading)
        reward += 0.1 * heading_error  # small reward for aligning with the waypoint direction




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



        # reward shaping: additional shaping terms accumulate into `reward`

        # distance to goal shaping
        old_dist = np.linalg.norm([self.old_x - self.goal[0], self.old_y - self.goal[1]])
        new_dist = np.linalg.norm([self.x - self.goal[0], self.y - self.goal[1]])
        reward += 3.0 * (old_dist - new_dist)  # reward for getting closer to the target

        # boulder proximity penalty (encourage the rover to stay away from boulders)
        self.d_edge, _ = self._compute_boulder_features()
        if self.d_edge < 2.0:  # if the rover is within 2 meters of a boulder edge
            reward -= (2.0 - self.d_edge) * 2.0  # penalty increases as the rover gets closer to the boulder

        # control effort penalty (encourage energy-efficient solutions)
        reward -= 0.001 * (u_R**2 + u_L**2)

        # time penalty (encourage faster solutions)
        reward -= 0.05

        # forward speed bonus (encourage the rover to keep moving forward)
        speed = abs(v)
        reward += 0.1 * speed

        # penalty for being to slow (encourage the rover to maintain a minimum speed)
        if speed < 0.05:
            reward -= 0.5

        # direction-consistent bonus (encourage the rover to maintain a consistent heading towards the target)
        self.goal_vector = np.array([self.goal[0] - self.x, self.goal[1] - self.y])
        self.heading_vector = np.array([math.cos(self.theta), math.sin(self.theta)])
        alignment = np.dot(self.goal_vector, self.heading_vector) / (np.linalg.norm(self.goal_vector) + 1e-6)

        # penalize when facing away from the target and reward when facing towards the target
        if alignment < 0:
            reward -= 0.1  # small penalty for facing away from the target

        # penalize when reversing direction (encourage the rover to maintain a consistent heading towards the target)
        if np.sign(v) != np.sign(self.prev_v):
            reward -= 0.2  # penalty for reversing direction
        self.prev_v = v

        # penalized for large angular velocity (encourage smoother trajectories)
        reward -= 0.05 * abs(omega)

        # penalized for rapid heading changes (encourage smoother trajectories)
        heading_change = abs(self.theta - getattr(self, "prev_theta", self.theta))
        reward -= 0.01 * heading_change
        self.prev_theta = self.theta

        # bonus for consistent forward movement (encourage the rover to maintain a consistent heading towards the target)\
        if v > 0:
            reward += 0.05 * alignment  # small bonus for facing towards the target when moving forward



        # termination conditions
        # target check
        if self._in_target_region():
            reward = 200.0
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
        for b in self.boulders:
            try:
                bx, by = b[0], b[1]
                br = b[2] if len(b) > 2 else self.R
            except Exception:
                bx, by = b
                br = self.R
            dist = np.hypot(self.x - bx, self.y - by)
            # safe distance depends on boulder radius
            if dist < (br + (np.sqrt(2) / 2) * self.W):
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

        nearest_br = self.R
        for b in self.boulders:
            try:
                bx, by = b[0], b[1]
                br = b[2] if len(b) > 2 else self.R
            except Exception:
                bx, by = b
                br = self.R
            vec = np.array([bx - self.x, by - self.y], dtype=np.float32)
            dist_center = np.linalg.norm(vec)

            if dist_center < nearest_dist:
                nearest_dist = dist_center
                nearest_vec = vec
                nearest_br = br

        # distance to boulder edge (use the nearest boulder's radius)
        d_edge = nearest_dist - nearest_br

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
        for b in self.boulders:
            try:
                bx, by = b[0], b[1]
                br = b[2] if len(b) > 2 else self.R
            except Exception:
                bx, by = b
                br = self.R
            pygame.draw.circle(
                self.screen,
                (128, 128, 128),  # gray boulders
                (int(bx * self.scale), int(by * self.scale)),
                int(br * self.scale)
            )

        # draw rover with a clear heading cue
        rover_size = int(self.W * self.scale)
        rover_surface = pygame.Surface((rover_size, rover_size), pygame.SRCALPHA)
        pygame.draw.rect(rover_surface, (0, 0, 255), rover_surface.get_rect())  # blue rover body

        center = (rover_size // 2, rover_size // 2)
        nose_length = rover_size // 2
        nose_end = (min(rover_size - 1, rover_size // 2 + nose_length), rover_size // 2)
        pygame.draw.line(rover_surface, (255, 255, 255), center, nose_end, 4)
        pygame.draw.polygon(
            rover_surface,
            (255, 215, 0),
            [
                (min(rover_size - 1, rover_size // 2 + nose_length), rover_size // 2),
                (max(0, rover_size // 2 + nose_length - 12), rover_size // 2 - 6),
                (max(0, rover_size // 2 + nose_length - 12), rover_size // 2 + 6),
            ],
        )

        rotated_rover = pygame.transform.rotate(rover_surface, -math.degrees(self.theta))
        rover_rect = rotated_rover.get_rect(center=(int(self.x * self.scale), int(self.y * self.scale)))
        self.screen.blit(rotated_rover, rover_rect)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])