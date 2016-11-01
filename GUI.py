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
EVOL_ITERATIONS		= 10

# Graphics parameters
WINDOW_TITLE		= "Rob538 Project - Rover Domain"
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
def init_canvas(roverDomain):
	global master_window
	global canvas
	canvas_width = roverDomain.width
	canvas_height = roverDomain.height
	master_window = Tk()
	master_window.title(WINDOW_TITLE)
	canvas = Canvas(master_window, width=canvas_width, height=canvas_height, background='#FFFFFF')
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
		obj = canvas.create_polygon(get_points_triangle(agent, l=ROVER_SIZE), fill=ROVER_COLOR)

	# Drawing the POIs
	for poi in roverDomain.poi_list:
		obj = canvas.create_polygon(get_points_triangle(poi, l=POI_SIZE), fill=POI_COLOR)

	# Updating the canvas
	canvas.update()

# =======================================================
# Simulation
# =======================================================

# Initialize world
def init_world():

	# Initializing rovers
	for i in range(NUM_ROVERS):
		roverDomain.add_rover(180 + 20*i, 200)

	# Initializing POIs
	for i in range(NUM_POIS):
		roverDomain.add_poi(random.randint(0, WORLD_HEIGHT), random.randint(0, WORLD_WIDTH))

	# Setting POIs initial velocities
	for poi in roverDomain.poi_list:
		poi.speed = random.randint(2, 8) / 20.0
		poi.heading = random.randint(0, WORLD_HEIGHT) / WORLD_HEIGHT * 2 * math.pi

# Takes a team of networks to evaluate rovers
# Each network is given a performance value at the end
def evaluate_rover_team(team, disp=True):
    avgiter = 1
    rewardsum = 0
    for n in range(avgiter):
        roverDomain.reset_agents(opts='RandomPR')
        for i in range(NUM_SIM_STEPS):
            for poi in roverDomain.poi_list:
                poi.walk()

            NN_index = 0
            for rover in roverDomain.rover_list:
                NN = team[NN_index]
                inputs = rover.return_NN_inputs()
                outputs = NN.forward(inputs)
                rover.walk(outputs)
                NN_index += 1

            if disp:
                draw_world(roverDomain)
        #rewardsum += calculate_reward(catch_time, steps) # REWARD STRUCTURES COME HERE!

    #NN.performance = rewardsum / avgiter


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
init_canvas(roverDomain)
init_world()

population = []

# INITIALIZE <POPULATION_SIZE> Neural Nets
# with 8 input, 2 output and 10 hidden units for every rover.
for rover in roverDomain.rover_list:
    for i in range(POPULATION_SIZE):
        NN = NeuralNet(NN_NUM_INPUT_LRS, NN_NUM_OUTPUT_LRS, NN_NUM_HIDDEN_LRS)
        rover.population.append(NN)

# EVOLUTION ================================================
generation_count = 0
for i in range(EVOL_ITERATIONS):

    # Mutate each NN to have 2k NNs
    for rover in roverDomain.rover_list:
        mutantlist = []
        for NN in rover.population:
            mutant = copy.deepcopy(NN)
            mutant.perturb_weights(PERTURBATION)
            mutantlist.append(mutant)
        rover.population += mutantlist

    # Randomly select one from each to form a team
    team = []
    for rover in roverDomain.rover_list:
        team.append(random.choice(rover.population)) # Same random!!!!

    # Evaluate Rover team, assess performance
    evaluate_rover_team(team)

    # Retain k best
    for rover in roverDomain.rover_list:
        rover.population = remove_worst(rover.population)

    # Increment Gen counter
    generation_count += 1
# ==========================================================

mainloop()

