from Tkinter import *
from Simulator import *
import pdb
import numpy as np
import pickle
import matplotlib.pyplot as plt
import datetime
import time
import csv



# =======================================================
# Parameters
# =======================================================


# NN Parameters
NN_NUM_INPUT_LRS	= 16
NN_NUM_OUTPUT_LRS	= 2
NN_NUM_HIDDEN_LRS	= 3

# Evolution parameters
POPULATION_SIZE		= 10
MUTATION_STD		= 0.5
NUM_GENERATIONS		= 200

# Graphics parameters
WINDOW_TITLE		= "Rob538 Project - Rover Domain"
BKG_COLOR			= 'white'
ROVER_COLOR			= 'orange'
POI_COLOR			= 'red'
ROVER_SIZE			= 5
POI_SIZE			= 3
SLEEP_VIEW			= 0.025
ENABLE_GRAPHICS		= 0 # Enabling graphics will load NNs from file
PLOT_ON             = 1

# World parameters
NUM_SIM_STEPS		= 100
WORLD_WIDTH			= 240.0
WORLD_HEIGHT		= 240.0
NUM_ROVERS			= 1
NUM_POIS			= 10
POI_MIN_VEL			= 0
POI_MAX_VEL			= 0
MIN_SENSOR_DIST		= 10
MAX_SENSOR_DIST		= 500

HOLONOMIC_ROVER		= 1
RND_START_EPISODE	= 1
RND_START_ALL		= 1
INPUT_SCALING		= 100
OUTPUT_SCALING		= 5

# File Parameters
NN_WEIGHTS_FILENAME	= "NN_"
RESULTS_FILENAME = "RS_%s_POI%s_ROV%s" %(str(datetime.datetime.now()), str(NUM_POIS), str(NUM_ROVERS))
RWD_FILENAME		= "RWD_"




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
	'MAX_SENSOR_DIST'	: MAX_SENSOR_DIST,

	'HOLONOMIC_ROVER'	: HOLONOMIC_ROVER,
	'RND_START_EPISODE'	: RND_START_EPISODE,
	'RND_START_ALL'		: RND_START_ALL,
	'INPUT_SCALING'		: INPUT_SCALING,
	'OUTPUT_SCALING'	: OUTPUT_SCALING,


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

if RND_START_ALL:
	simulator.init_world(POI_MIN_VEL, POI_MAX_VEL)
else:
	simulator.init_world_custom(POI_MIN_VEL, POI_MAX_VEL, POI_LOCATIONS, ROVER_LOCATIONS)

simulator.initRoverNNs(POPULATION_SIZE, NN_NUM_INPUT_LRS, NN_NUM_OUTPUT_LRS, NN_NUM_HIDDEN_LRS, HOLONOMIC_ROVER)


best_perf = np.zeros([NUM_ROVERS, NUM_GENERATIONS])
avg_perf  = np.zeros([NUM_ROVERS, NUM_GENERATIONS])
std_perf  = np.zeros([NUM_ROVERS, NUM_GENERATIONS])




if ENABLE_GRAPHICS: # Visualizing results

	# Loading best weights for each robot
	simulator.load_bestWeights(NN_WEIGHTS_FILENAME)

	# Running NUM_GENERATIONS times
	for i in range(0,NUM_GENERATIONS,10):
		execute_episode(0)

else:	# Evolving new NNs
	if PLOT_ON:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		Ln, = ax.plot([0],'ro')
		plt.ion()
		plt.show()

	# Cleaning the history file
	for j in range(NUM_ROVERS):
		file = open(RWD_FILENAME+str(j),'w')
		file.write("")
		file.close()

	generation_count = 0
	mutation_Update = MUTATION_STD
	for i in range(NUM_GENERATIONS):
		mutation_Update = max(0.99 * mutation_Update, 0.01)
		# Generate twice as many NNs doing mutated copies
		simulator.mutateNNs(mutation_Update)

		# Running an episode for each population member
		for j in range(2*POPULATION_SIZE):
			# print "Generation %d, Population set %d" % (generation_count, i)
			execute_episode(j)

		# Printing overall performance of each NN for each rover
		for j in range(NUM_ROVERS):
			print "Generation %d - Rover %d:" %(generation_count, j),
			performance_list = simulator.get_performance(j)
			performance_list.sort()
			if PLOT_ON:
				for p in performance_list:
					print "%.3f " % p,
				print ""
				print("Best Performer Score: %s" %max(performance_list))
				print("Average Performer Score: %s" %np.mean(performance_list))
				print("Standard Deviation p: %s" %np.std(performance_list))
				
			best_perf[j][i] = max(performance_list)
			avg_perf[j][i] = np.mean(performance_list)
			std_perf[j][i] = np.std(performance_list)/np.mean(performance_list)
		if PLOT_ON:
			for j in range(NUM_ROVERS):
				plt.plot(np.arange(i+1),best_perf[j][0:i+1], 'o', color = (mutation_Update, 1-mutation_Update, 0) )
				plt.pause(.001)
		
			# Writing to the history file
			file = open(RWD_FILENAME+str(j),'a')
			wr = csv.writer(file)
			wr.writerow(performance_list)
			file.close()


		if i%(NUM_GENERATIONS/100) == 0:
			with open(RESULTS_FILENAME + '.pickle', 'w') as f:
				pickle.dump([best_perf, avg_perf, std_perf, InputParameters], f)



		simulator.store_bestWeights(NN_WEIGHTS_FILENAME)

		simulator.select()

		generation_count += 1
