from Tkinter import *
from Simulator import *
import sys
import csv
import time

# =======================================================
# Parameters
# =======================================================

# File parameters
NN_WEIGHTS_FILENAME	= "NN_"
RWD_FILENAME		= "RWD_"

# Reward parameters
LOCAL_REWARD		= 0
GLOBAL_REWARD		= 1
DIFF_REWARD			= 2
RWD_TYPE			= LOCAL_REWARD
RWD_PATH			= ["LocalRwd", "GlobalRwd", "DiffRwd"]

NN_WEIGHTS_FILENAME = RWD_PATH[RWD_TYPE]+"/"+NN_WEIGHTS_FILENAME
RWD_FILENAME		= RWD_PATH[RWD_TYPE]+"/"+RWD_FILENAME

# NN parameters
ENABLE_VEL_SENSING	= 1
NN_IN_LYR_SIZE		= 12 if ENABLE_VEL_SENSING else 8
NN_OUT_LYR_SIZE		= 2
NN_HID_LYR_SIZE		= 3

# Evolution parameters
POPULATION_SIZE		= 10
MUTATION_STD		= 0.15
NUM_GENERATIONS		= 100000

# Graphics parameters
WINDOW_TITLE		= "Rob538 Project - Rover Domain"
BKG_COLOR			= 'white'
ROVER_COLOR			= 'orange'
POI_COLOR			= 'red'
ROVER_SIZE			= 5
POI_SIZE			= 3
SLEEP_VIEW			= 0.100

# World parameters
NUM_SIM_STEPS		= 50
WORLD_WIDTH			= 200.0
WORLD_HEIGHT		= 200.0
NUM_ROVERS			= 5
NUM_POIS			= 5
POI_MIN_VEL			= 0.0
POI_MAX_VEL			= 1.0
MIN_SENSOR_DIST		= 5
MAX_SENSOR_DIST		= 500

HOLONOMIC_ROVER		= 1
RND_START_EPISODE	= 1
RND_START_ALL		= 1
INPUT_SCALING		= 100
OUTPUT_SCALING		= 5
STEERING_ONLY		= -5.0

# Enabling/disabling evolution with command line parameter
if len(sys.argv) > 1 and sys.argv[1][0] == '-' and sys.argv[1][1] == 'e':
	print "Evolution enabled. No graphics."
	disable_evol = 0
else:
	print "Evolution disabled. Graphics activated."
	disable_evol = 1

# For custom agent initialization
POI_LOCATIONS = [(25,	 25,	1),
			 	 (120,	 25,	1),
				 (80,	100,	2),
				 (50,	120,	3)]

ROVER_LOCATIONS = [(120, 120, 0)]

# =======================================================
# Graphics
# =======================================================


# Initialize the graphics canvas
def init_canvas():
	global master_window
	global canvas
	master_window = Tk()
	master_window.title(WINDOW_TITLE)
	canvas = Canvas(master_window, width=WORLD_WIDTH, height=WORLD_HEIGHT, background=BKG_COLOR)
	canvas.pack()

# Points for drawing agent's body
def get_points_triangle(agent, l=5):
	x = agent.pos[0]
	y = agent.pos[1]
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
		canvas.create_polygon(get_points_triangle(poi, l=POI_SIZE), fill=POI_COLOR)

	# Updating the canvas
	canvas.update()

# =======================================================
# Simulation
# =======================================================

# Episode execution
def execute_episode(pop_set):

	# Randomizing starting positions
	simulator.reset_agents(RND_START_EPISODE, RND_START_EPISODE)

	# Reset performance counter
	simulator.reset_performance(pop_set)

	# Running through each simulation step
	for i in range(NUM_SIM_STEPS):
		simulator.sim_step(pop_set, STEERING_ONLY)
		if disable_evol:
			draw_world(simulator)
			time.sleep(SLEEP_VIEW)

	# Computing reward
	simulator.compute_global_reward(pop_set)
	if RWD_TYPE == LOCAL_REWARD:
		simulator.local_reward(pop_set)
	if RWD_TYPE == GLOBAL_REWARD:
		simulator.global_reward(pop_set)
	if RWD_TYPE == DIFF_REWARD:
		simulator.diff_reward(pop_set)


# =======================================================
# Main code
# =======================================================

if disable_evol:
	init_canvas()

simulator = Simulator(
		min_sensor_dist 	= MIN_SENSOR_DIST,
		max_sensor_dist 	= MAX_SENSOR_DIST,
		num_rovers 			= NUM_ROVERS,
		num_pois 			= NUM_POIS,
		world_width 		= WORLD_WIDTH,
		world_height 		= WORLD_HEIGHT)

if RND_START_ALL:
	simulator.init_world(POI_MIN_VEL, POI_MAX_VEL, HOLONOMIC_ROVER)
else:
	simulator.init_world_custom(POI_MIN_VEL, POI_MAX_VEL, POI_LOCATIONS, ROVER_LOCATIONS)

simulator.initRoverNNs(POPULATION_SIZE, NN_IN_LYR_SIZE, NN_OUT_LYR_SIZE, NN_HID_LYR_SIZE, INPUT_SCALING, OUTPUT_SCALING)

if disable_evol: # Visualizing results

	# Loading best weights for each robot
	simulator.load_bestWeights(NN_WEIGHTS_FILENAME)

	# Running NUM_GENERATIONS times
	for i in range(NUM_GENERATIONS):
		execute_episode(0)

else:	# Evolving new NNs

	# Cleaning the history files
	for j in range(NUM_ROVERS):
		file = open(RWD_FILENAME+str(j),'w')
		file.write("")
		file.close()
	file = open(RWD_FILENAME+'G','w')
	file.write("")
	file.close()

	generation_count = 0
	for i in range(NUM_GENERATIONS):
		
		print "Generation %d" % generation_count

		# Generate twice as many NNs doing mutated copies
		simulator.mutateNNs(MUTATION_STD)

		# Running an episode for each population member
		for j in range(2*POPULATION_SIZE):
			execute_episode(j)

		# Selecting best weights
		simulator.select()

		# Storing NNs weights for later execution/visualization
		simulator.store_bestWeights(NN_WEIGHTS_FILENAME)

		# Printing overall performance of each NN for each rover
		for j in range(NUM_ROVERS):
			performance_list = simulator.get_performance(j)
			print "Rover %d:" % j,
			for p in performance_list:
				print "%.3f " % p,
			print ""

			# Writing to the history file
			file = open(RWD_FILENAME+str(j),'a')
			wr = csv.writer(file)
			wr.writerow(performance_list)
			file.close()

		# Writing global reward to the history file
		global_rwd_list = []
		for j in range(POPULATION_SIZE):
			global_rwd_list.append(simulator.global_rwd)
		file = open(RWD_FILENAME+'G','a')
		wr = csv.writer(file)
		wr.writerow(global_rwd_list)
		file.close()

		generation_count += 1
