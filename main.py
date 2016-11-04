from Tkinter import *
from Simulator import *
import pdb
import numpy as np
import pickle
import matplotlib.pyplot as plt
import datetime



# =======================================================
# Parameters
# =======================================================


# NN Parameters
NN_NUM_INPUT_LRS	= 16
NN_NUM_OUTPUT_LRS	= 2
NN_NUM_HIDDEN_LRS	= 3

# Evolution parameters
POPULATION_SIZE		= 10
MUTATION_STD		= 0.05
NUM_GENERATIONS		= 1000

# Graphics parameters
WINDOW_TITLE		= "Rob538 Project - Rover Domain"
BKG_COLOR			= 'white'
ROVER_COLOR			= 'orange'
POI_COLOR			= 'red'
ROVER_SIZE			= 5
POI_SIZE			= 3
ENABLE_GRAPHICS		= 1

# World parameters
NUM_SIM_STEPS		= 500
WORLD_WIDTH			= 240.0
WORLD_HEIGHT		= 240.0
NUM_ROVERS			= 1
NUM_POIS			= 10
POI_MIN_VEL			= 0
POI_MAX_VEL			= 0
MIN_SENSOR_DIST		= 10
MAX_SENSOR_DIST		= 2000


# File Parameters
NN_WEIGHTS_FILENAME	= "NN_"
RESULTS_FILENAME = "RS_%s_POI%s_ROV%s" %(str(datetime.datetime.now()), str(NUM_POIS), str(NUM_ROVERS))
###make dictionary with all variables!!!
InputParameters = {
	# NN Parameters
	'NN_NUM_INPUT_LRS'	: NN_NUM_INPUT_LRS,
	'NN_NUM_OUTPUT_LRS'	: NN_NUM_OUTPUT_LRS,
	'NN_NUM_HIDDEN_LRS'	: NN_NUM_HIDDEN_LRS,

	# Evolution parameters
	'POPULATION_SIZE'	: POPULATION_SIZE,
	'MUTATION_STD'		: MUTATION_STD,
	'NUM_GENERATIONS'	: NUM_GENERATIONS,

	# World parameters
	'NUM_SIM_STEPS'		: NUM_SIM_STEPS,
	'WORLD_WIDTH'		: WORLD_WIDTH,
	'WORLD_HEIGHT'		: WORLD_HEIGHT,
	'NUM_ROVERS'		: NUM_ROVERS,
	'NUM_POIS'			: NUM_POIS,
	'POI_MIN_VEL'		: POI_MIN_VEL,
	'POI_MAX_VEL'		: POI_MAX_VEL,
	'MIN_SENSOR_DIST'	: MIN_SENSOR_DIST,
	'MAX_SENSOR_DIST'	: MAX_SENSOR_DIST


}



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
	st = math.sin(t)
	ct = math.cos(t)
	p1 = [x + l * st, y - l * ct]
	p2 = [x - l * st, y + l * ct]
	p3 = [x + 3 * l * ct, y + 3 * l * st]
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

	# pdb.set_trace()

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


best_perf = np.zeros(NUM_GENERATIONS)
avg_perf = np.zeros(NUM_GENERATIONS)
std_perf = np.zeros(NUM_GENERATIONS)




if ENABLE_GRAPHICS:
	simulator.load_bestWeights(NN_WEIGHTS_FILENAME)
	for i in range(NUM_GENERATIONS):
		execute_episode(0)
# if ENABLE_GRAPHICS or ~ENABLE_GRAPHICS:
# if ~ENABLE_GRAPHICS:
else:
	fig = plt.figure()
	ax = fig.add_subplot(111)
	Ln, = ax.plot([0,0],'ro')
	plt.ion()
	plt.show()
	generation_count = 0
	plt.ion()
	for i in range(NUM_GENERATIONS):

		# Generate twice as many NNs doing mutated copies
		simulator.mutateNNs(MUTATION_STD)

		# Running an episode for each population member
		for j in range(2*POPULATION_SIZE):
			# print "Generation %d, Population set %d" % (generation_count, i)
			execute_episode(j)

		# Printing overall performance of each NN for each rover
		for j in range(NUM_ROVERS):
			print "Generation %d - Rover %d:" %(generation_count, j),
			performance_list = simulator.get_performance(j)
			performance_list.sort()
			for p in performance_list:
				print "%.3f " % p,
			print ""
			print("Best Performer Score: %s" %max(performance_list))
			print("Average Performer Score: %s" %np.mean(performance_list))
			print("Standard Deviation p: %s" %np.std(performance_list))
			
		best_perf[i] = max(performance_list)
		avg_perf[i] = np.mean(performance_list)
		std_perf[i] = np.std(performance_list)/np.mean(performance_list)
		plt.plot(np.arange(i+1),best_perf[0:i+1], 'ro')
		plt.pause(.001)
		
		with open(RESULTS_FILENAME + '.pickle', 'w') as f:
			pickle.dump([best_perf, avg_perf, std_perf, InputParameters], f)



		simulator.store_bestWeights(NN_WEIGHTS_FILENAME)

		simulator.select()

		generation_count += 1
