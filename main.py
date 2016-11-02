from Tkinter import *
from Simulator import *

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
POPULATION_SIZE		= 3
MUTATION_STD		= 0.25
NUM_GENERATIONS		= 10

# Graphics parameters
WINDOW_TITLE		= "Rob538 Project - Rover Domain"
BKG_COLOR			= 'white'
ROVER_COLOR			= 'orange'
POI_COLOR			= 'red'
ROVER_SIZE			= 5
POI_SIZE			= 3
ENABLE_GRAPHICS		= 1

# World parameters
NUM_SIM_STEPS		= 1000
WORLD_WIDTH			= 240.0
WORLD_HEIGHT		= 240.0
NUM_ROVERS			= 1
NUM_POIS			= 10
POI_MIN_VEL			= 0.01
POI_MAX_VEL			= 0.05
MIN_SENSOR_DIST		= 10
MAX_SENSOR_DIST		= 500

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
	simulator.reset_agents()

	# Reset performance counter
	simulator.reset_performance(pop_set)

	# Running through each simulation step
	for i in range(NUM_SIM_STEPS):
		simulator.sim_step(pop_set)
		if ENABLE_GRAPHICS:
			draw_world(simulator)


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

simulator.init_world(POI_MIN_VEL, POI_MAX_VEL)

simulator.initRoverNNs(POPULATION_SIZE, NN_NUM_INPUT_LRS, NN_NUM_OUTPUT_LRS, NN_NUM_HIDDEN_LRS)

if ENABLE_GRAPHICS:
	simulator.load_bestWeights(NN_WEIGHTS_FILENAME)
	for i in range(NUM_GENERATIONS):
		execute_episode(0)
else:

	generation_count = 0
	for i in range(NUM_GENERATIONS):

		# Generate twice as many NNs doing mutated copies
		simulator.mutateNNs(MUTATION_STD)

		# Running an episode for each population member
		for j in range(2*POPULATION_SIZE):
			# print "Generation %d, Population set %d" % (generation_count, i)
			execute_episode(j)

		# Printing overall performance of each NN for each rover
		for j in range(NUM_ROVERS):
			print "Rover %d:" % j,
			performance_list = simulator.get_performance(j)
			performance_list.sort()
			for p in performance_list:
				print "%.3f " % p,
			print ""

		simulator.store_bestWeights(NN_WEIGHTS_FILENAME)

		simulator.select()

		generation_count += 1
