from Tkinter import *
from Simulator import *
import os
import sys
import csv
import time
import math


import numpy as np
import pdb

# =======================================================
# Parameters
# =======================================================

# Reward parameters
LOCAL_REWARD		= 0
GLOBAL_REWARD		= 1
DIFF_REWARD			= 2
RWD_PATH			= ["LocalRwd", "GlobalRwd", "DiffRwd"]

# File parameters
NN_WEIGHTS_PREFX	= "nn_weights/NN_"
RWD_FILENAME		= "SYS_RWD"

# NN parameters
ENABLE_VEL_SENSING	= 0
NN_IN_LYR_SIZE		= 12 if ENABLE_VEL_SENSING else 8
NN_OUT_LYR_SIZE		= 2
NN_HID_LYR_SIZE		= 10

# Evolution parameters
POPULATION_SIZE		= 25
MUTATION_STD		= 0.10
NUM_GENERATIONS		= 200
# REMOVE_RATIO		= 0.01
# NUM_CHILDREN		= int(REMOVE_RATIO*POPULATION_SIZE)
NUM_CHILDREN		= 5

# Graphics parameters
WINDOW_TITLE		= "Rob538 Project - Rover Domain"
BKG_COLOR			= 'white'
ROVER_COLOR			= 'orange'	
POI_COLOR			= 'red'
ROVER_SIZE			= 10
POI_SIZE			= 3
SLEEP_VIEW			= 0.100
ZOOM				= 8.0

# World parameters
NUM_SIM_STEPS		= 15
WORLD_WIDTH			= 115.0
WORLD_HEIGHT		= 100.0
NUM_ROVERS			= 20
NUM_POIS			= 7**2
POI_MIN_VEL			= 0.0
POI_MAX_VEL			= 0.0
MIN_SENSOR_DIST		= 1
MAX_SENSOR_DIST		= 500
GRID_SIZE			= 10

HOLONOMIC_ROVER		= 1
MAX_TRAVEL_STEP		= 3
RND_START_EPISODE	= 0
RND_START			= 0
RESAMPLE_POIS		= 0
RND_CUSTOM			= 0.05*GRID_SIZE
OUTPUT_SCALING		= 1
STEERING_ONLY		= -5.0

SELECTION_METHOD 	= 'HOF'  #'Team', 'HOF'
# HOF Selection
best_pop			= np.ones(NUM_ROVERS, dtype = int) * -1

# =======================================================
# Command Line parameters
# =======================================================

input_scaling		= MIN_SENSOR_DIST**2

# Choosing reward structure
rwd_type = LOCAL_REWARD
if len(sys.argv) > 1 and sys.argv[1][0] == '-':
	if sys.argv[1][1] == 'L':
		rwd_type = LOCAL_REWARD
		print "LOCAL_REWARD."
	elif sys.argv[1][1] == 'G':
		rwd_type = GLOBAL_REWARD
		print "GLOBAL_REWARD."
	elif sys.argv[1][1] == 'D':
		rwd_type = DIFF_REWARD
		print "DIFF_REWARD."

# Enabling/disabling evolution with command line parameter
if len(sys.argv) > 2 and sys.argv[2][0] == '-' and sys.argv[2][1] == 'e':
	print "Evolution enabled. No graphics."
	disable_evol = 0
else:
	print "Evolution disabled. Graphics activated."
	disable_evol = 1

# Evolution history file number
if len(sys.argv) > 3:
	RWD_FILENAME += sys.argv[3]

if len(sys.argv) > 4 and sys.argv[4][0] == '-':
	if sys.argv[4][1:].upper() == 'TEAM':
		SELECTION_METHOD = 'TEAM'
	elif sys.argv[4][1:].upper() == 'HOF':
		SELECTION_METHOD = 'HOF'

# =======================================================
# Parameter initialization
# =======================================================

# File Paths
nn_weights_path	= RWD_PATH[rwd_type]+"/"+NN_WEIGHTS_PREFX
rwd_hist_path	= RWD_PATH[rwd_type]+"/"+RWD_FILENAME

# For custom agent initialization
poi_init_pos = []
num_pois_side = int(math.floor(math.sqrt(NUM_POIS)))
for i in range(num_pois_side):
	for j in range(num_pois_side):
		pos_i = GRID_SIZE*i + WORLD_WIDTH/2 - GRID_SIZE*num_pois_side/2 + GRID_SIZE/2
		pos_j = GRID_SIZE*j + WORLD_HEIGHT/2 - GRID_SIZE*num_pois_side/2 + GRID_SIZE/2
		poi_init_pos.append((pos_i,pos_j,0))
rover_init_pos = [(WORLD_WIDTH/2, WORLD_HEIGHT/2, 0)]

# =======================================================
# Graphics
# =======================================================


# Initialize the graphics canvas
def init_canvas():
	global master_window
	global canvas
	master_window = Tk()
	master_window.title(WINDOW_TITLE)
	canvas = Canvas(master_window, width=ZOOM*WORLD_WIDTH, height=ZOOM*WORLD_HEIGHT, background=BKG_COLOR)
	canvas.pack()

# Points for drawing agent's body
def get_points_triangle(agent, l=5):
	x = ZOOM*agent.pos[0]
	y = ZOOM*agent.pos[1]
	t = agent.heading
	p1 = [x + l * math.sin(t), y - l * math.cos(t)]
	p2 = [x - l * math.sin(t), y + l * math.cos(t)]
	p3 = [x + 3 * l * math.cos(t), y + 3 * l * math.sin(t)]
	return [p1,p2,p3]

# Draw world
def draw_world(simulator):

	# Clearing the drawing canvas
	canvas.delete("all")

	# Drawing the rovers
	for agent in simulator.rover_list:
		canvas.create_polygon(get_points_triangle(agent, l=ROVER_SIZE), fill=ROVER_COLOR)

	# Drawing the POIs
	for poi in simulator.poi_list:
		canvas.create_polygon(get_points_triangle(poi, l=POI_SIZE), fill="#0%x0"%(15*(MIN_SENSOR_DIST**2)*max(poi.obs)))

	# Updating the canvas
	canvas.update()

# =======================================================
# Simulation
# =======================================================

# Episode execution
def execute_episode(pop_set):

	# Randomizing starting positions
	simulator.reset_agents(RND_START_EPISODE, RND_START_EPISODE, RND_CUSTOM, RESAMPLE_POIS)

	# Reset performance counter
	simulator.reset_performance(pop_set)

	# Running through each simulation step
	for i in range(NUM_SIM_STEPS):
		simulator.sim_step(pop_set, STEERING_ONLY, MAX_TRAVEL_STEP, ENABLE_VEL_SENSING)
		if disable_evol:
			draw_world(simulator)
			time.sleep(SLEEP_VIEW)

	# Computing reward
	simulator.compute_global_reward()
	if rwd_type == LOCAL_REWARD:
		simulator.local_reward(pop_set)
	if rwd_type == GLOBAL_REWARD:
		simulator.global_reward(pop_set)
	if rwd_type == DIFF_REWARD:
		simulator.diff_reward(pop_set)


# =======================================================
# Main code
# =======================================================

if disable_evol:
	init_canvas()

simulator = Simulator(
		min_sensor_dist_sqr	= MIN_SENSOR_DIST**2,
		max_sensor_dist 	= MAX_SENSOR_DIST,
		num_rovers 			= NUM_ROVERS,
		num_pois 			= NUM_POIS,
		world_width 		= WORLD_WIDTH,
		world_height 		= WORLD_HEIGHT)

if RND_START:
	simulator.init_world(POI_MIN_VEL, POI_MAX_VEL, HOLONOMIC_ROVER)
else:
	simulator.init_world_custom(POI_MIN_VEL, POI_MAX_VEL, poi_init_pos, rover_init_pos)

simulator.initRoverNNs(POPULATION_SIZE, NN_IN_LYR_SIZE, NN_OUT_LYR_SIZE, NN_HID_LYR_SIZE, input_scaling, OUTPUT_SCALING)

if disable_evol: # Visualizing results

	# Loading best weights for each robot
	simulator.load_bestWeights(nn_weights_path)

	# Running NUM_GENERATIONS times
	for i in range(NUM_GENERATIONS):
		pop_set = np.zeros(NUM_ROVERS, dtype = int)
		execute_episode(pop_set)

else:	# Evolving new NNs

	# Cleaning the history files
	os.system("rm "+rwd_hist_path)
	os.system("rm "+RWD_PATH[rwd_type]+"/nn_weights/*")

	generation_count = 0
	for i in range(NUM_GENERATIONS):
		
		print "Generation %d" % generation_count

		# Running an episode for each population member
		global_rwd_hist = []


		if SELECTION_METHOD == 'HOF':
			for r in range(NUM_ROVERS):
				pop_set = copy.deepcopy(best_pop)
				for j in range(POPULATION_SIZE):
					pop_set[r] = j
					execute_episode(pop_set)
					global_rwd_hist.append(simulator.global_rwd)

		elif SELECTION_METHOD == 'TEAM':
			for j in range(POPULATION_SIZE):
				pop_set = np.ones(NUM_ROVERS, dtype = int)*j
				execute_episode(pop_set)
				global_rwd_hist.append(simulator.global_rwd)

		else:
			for j in range(POPULATION_SIZE):
				pop_set = np.ones(NUM_ROVERS, dtype = int)*j
				execute_episode(pop_set)
				global_rwd_hist.append(simulator.global_rwd)


		# Writing global reward to the history file
		file = open(rwd_hist_path,'a')
		wr = csv.writer(file)
		wr.writerow(global_rwd_hist)
		file.close()

		# Storing NNs weights for later execution/visualization
		simulator.store_bestWeights(nn_weights_path)

		# Selecting best weights
		simulator.select(NUM_CHILDREN)

		# Generate twice as many NNs doing mutated copies
		simulator.gen_children(MUTATION_STD, NUM_CHILDREN)

		generation_count += 1
