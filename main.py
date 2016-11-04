from Tkinter import *
from Simulator import *
import time

# =======================================================
# Parameters
# =======================================================

# File Parameters
NN_WEIGHTS_FILENAME	= "NN_"

# NN Parameters
NN_NUM_INPUT_LRS	= 8
NN_NUM_OUTPUT_LRS	= 2
NN_NUM_HIDDEN_LRS	= 3

# Evolution parameters
POPULATION_SIZE		= 10
MUTATION_STD		= 0.15
NUM_GENERATIONS		= 100

# Graphics parameters
WINDOW_TITLE		= "Rob538 Project - Rover Domain"
BKG_COLOR			= 'white'
ROVER_COLOR			= 'orange'
POI_COLOR			= 'red'
ROVER_SIZE			= 5
POI_SIZE			= 3
SLEEP_VIEW			= 0.025
ENABLE_GRAPHICS		= 1	# Enabling graphics will load NNs from file

# World parameters
NUM_SIM_STEPS		= 400
WORLD_WIDTH			= 240.0
WORLD_HEIGHT		= 240.0
NUM_ROVERS			= 1
NUM_POIS			= 2
POI_MIN_VEL			= 0.0
POI_MAX_VEL			= 0.0
MIN_SENSOR_DIST		= 10
MAX_SENSOR_DIST		= 500
INPUT_SCALING 		= 100

HOLONOMIC_ROVER		= 0
RND_START_EPISODE	= 0
RND_START_ALL		= 0

# For custom agent initialization
POI_LOCATIONS = [(25,	 25,	1),
			 	 (180,	 75,	1),
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
	simulator.reset_agents(RND_START_EPISODE, RND_START_EPISODE, RND_START_EPISODE and not HOLONOMIC_ROVER)

	# Reset performance counter
	simulator.reset_performance(pop_set)

	# Running through each simulation step
	for i in range(NUM_SIM_STEPS):
		simulator.sim_step(pop_set)
		if ENABLE_GRAPHICS:
			draw_world(simulator)
			time.sleep(SLEEP_VIEW)


# =======================================================
# Main code
# =======================================================

if ENABLE_GRAPHICS:
	init_canvas()

simulator = Simulator(
		min_sensor_dist 	= MIN_SENSOR_DIST,
		max_sensor_dist 	= MAX_SENSOR_DIST,
		num_rovers 			= NUM_ROVERS,
		num_pois 			= NUM_POIS,
		world_width 		= WORLD_WIDTH,
		world_height 		= WORLD_HEIGHT)

if RND_START_ALL:
	simulator.init_world(POI_MIN_VEL, POI_MAX_VEL)
else:
	simulator.init_world_custom(POI_MIN_VEL, POI_MAX_VEL, POI_LOCATIONS, ROVER_LOCATIONS)

simulator.initRoverNNs(POPULATION_SIZE, NN_NUM_INPUT_LRS, NN_NUM_OUTPUT_LRS, NN_NUM_HIDDEN_LRS, HOLONOMIC_ROVER)

if ENABLE_GRAPHICS: # Visualizing results

	# Loading best weights for each robot
	simulator.load_bestWeights(NN_WEIGHTS_FILENAME)

	# Running NUM_GENERATIONS times
	for i in range(NUM_GENERATIONS):
		execute_episode(0)

else:	# Evolving new NNs

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

		generation_count += 1
