"""Environment package constants."""

wr = 0.25 # Wheel radius (m)
rW = 2.0 # rover width (m)
max_time = 200.0 # max episode time (s)
sim_time = 0.0 # current simulation time (s)
prev_theta = 0.0 # previous heading angle (rad)

# for planner-related constants
grid_res = 0.5 # grid resolution (m)
grid = None # grid map (2D numpy array)
path = [] # planned path (list of (x, y) tuples)
current_waypoint_index = 0 # index of the current waypoint in the path