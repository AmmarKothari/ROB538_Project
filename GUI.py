from Tkinter import *
from RoverDomain import *
from NN_Unsupervised import NeuralNet
from operator import attrgetter
import time, copy
import numpy as np

# =======================================================
# Parameters
# =======================================================

# NN Parameters
NN_NUM_INPUT_LRS	= 8
NN_NUM_OUTPUT_LRS	= 2
NN_NUM_HIDDEN_LRS	= 10

# Evolution parameters
POPULATION_SIZE		= 10
PERTURBATION		= 0.25
NUM_EPISODES		= 10

# Graphics parameters
WINDOW_TITLE		= "Rob538 Project - Rover Domain"
BKG_COLOR			= 'white'
ROVER_COLOR			= 'orange'
POI_COLOR			= 'red'
ROVER_SIZE			= 5
POI_SIZE			= 3

# World parameters
NUM_SIM_STEPS		= 1000
WORLD_WIDTH			= 640.0
WORLD_HEIGHT		= 480.0
NUM_ROVERS			= 5
NUM_POIS			= 8

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
def draw_world(roverDomain):

	# Clearing the drawing canvas
	canvas.delete("all")

	# Drawing the rovers
	for agent in roverDomain.rover_list:
		canvas.create_polygon(get_points_triangle(agent, l=ROVER_SIZE), fill=ROVER_COLOR)

	# Drawing the POIs
	for poi in roverDomain.poi_list:
		canvas.create_polygon(get_points_triangle(poi, l=POI_SIZE), fill=POI_COLOR)

	# Updating the canvas
	canvas.update()

# =======================================================
# Simulation
# =======================================================

# Initialize world
def init_world():

	# Initializing rovers
	for i in range(NUM_ROVERS):
		roverDomain.add_rover(random.randint(0, WORLD_HEIGHT), random.randint(0, WORLD_WIDTH), random.randint(0, 360))

	# Initializing POIs
	for i in range(NUM_POIS):
		roverDomain.add_poi(random.randint(0, WORLD_HEIGHT), random.randint(0, WORLD_WIDTH), random.randint(0, 360))

	# Setting POIs initial velocities
	for poi in roverDomain.poi_list:
		poi.heading = random.randint(0, WORLD_HEIGHT) / WORLD_HEIGHT * 2 * math.pi
		poi.set_vel_lin(random.randint(2, 8) / 20.0)

# Takes a team of networks to evaluate rovers
# Each network is given a performance value at the end
def execute_episode():

	# Randomizing starting positions
	roverDomain.reset_agents()

	# Running through each simulation step
	for i in range(NUM_SIM_STEPS):

		sim_step()

		draw_world(roverDomain)

		# NN_index = 0
		# for rover in roverDomain.rover_list:
		# 	NN = team[NN_index]
		# 	inputs = rover.return_NN_inputs()
		# 	outputs = NN.forward(inputs)
		# 	rover.step(outputs)
		# 	NN_index += 1

# Iterate the world simulation
def sim_step():

	# POIs step
	for poi in roverDomain.poi_list:
		poi.sim_step()

	# Rovers step
	for rover in roverDomain.rover_list:
		rover.sim_step()


# Removing the worst performing rover
def remove_worst(NNlist, k=None):
	if k == None:
		k = len(NNlist)/2
	for i in range(k):
		random.shuffle(NNlist)
		worst = min(NNlist, key=attrgetter('performance'))
		NNlist.remove(worst)
	return NNlist

roverDomain = RoverDomain(WORLD_HEIGHT,WORLD_WIDTH)
init_canvas()
init_world()

population = []

# INITIALIZE <POPULATION_SIZE> Neural Nets
for rover in roverDomain.rover_list:
    for i in range(POPULATION_SIZE):
        NN = NeuralNet(NN_NUM_INPUT_LRS, NN_NUM_OUTPUT_LRS, NN_NUM_HIDDEN_LRS)
        rover.population.append(NN)

# EVOLUTION ================================================
episode_count = 0
for i in range(NUM_EPISODES):
	print "Episode %d" % episode_count

	# # Mutate each NN to have 2k NNs
	# for rover in roverDomain.rover_list:
	# 	mutantlist = []
	# 	for NN in rover.population:
	# 		mutant = copy.deepcopy(NN)
	# 		mutant.perturb_weights(PERTURBATION)
	# 		mutantlist.append(mutant)
	# 	rover.population += mutantlist

	# Randomly select one from each to form a team
	team = []
	for rover in roverDomain.rover_list:
		team.append(random.choice(rover.population)) # Same random!!!!

	# Evaluate Rover team, assess performance
	execute_episode()

	# # Retain k best
	# for rover in roverDomain.rover_list:
	# 	rover.population = remove_worst(rover.population)

	# Increment Gen counter
	episode_count += 1


# ==========================================================
