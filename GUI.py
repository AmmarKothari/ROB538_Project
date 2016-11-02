from Tkinter import *
from RoverDomain import *
import time
import numpy as np

# =======================================================
# Parameters
# =======================================================

# NN Parameters
NN_NUM_INPUT_LRS	= 8
NN_NUM_OUTPUT_LRS	= 2
NN_NUM_HIDDEN_LRS	= 10

# Evolution parameters
POPULATION_SIZE		= 15
PERTURBATION		= 0.25
NUM_EPISODES		= 10

# Graphics parameters
WINDOW_TITLE		= "Rob538 Project - Rover Domain"
BKG_COLOR			= 'white'
ROVER_COLOR			= 'orange'
POI_COLOR			= 'red'
ROVER_SIZE			= 5
POI_SIZE			= 3
DRAW				= 1

# World parameters
NUM_SIM_STEPS		= 1000
WORLD_WIDTH			= 640.0
WORLD_HEIGHT		= 480.0
NUM_ROVERS			= 5
NUM_POIS			= 8
POI_MIN_VEL			= 0.1
POI_MAX_VEL			= 0.4
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

# Takes a team of networks to evaluate rovers
# Each network is given a performance value at the end
def execute_episode():

	# Randomizing starting positions
	roverDomain.reset_agents()

	# Running through each simulation step
	for i in range(NUM_SIM_STEPS):
		roverDomain.sim_step()
		if DRAW:
			draw_world(roverDomain)


# =======================================================
# Main code
# =======================================================

init_canvas()
roverDomain = RoverDomain(MIN_SENSOR_DIST, MAX_SENSOR_DIST, NUM_ROVERS, NUM_POIS, WORLD_WIDTH, WORLD_HEIGHT)
roverDomain.init_world(POI_MIN_VEL, POI_MAX_VEL)
roverDomain.initRoverNNs(POPULATION_SIZE, NN_NUM_INPUT_LRS, NN_NUM_OUTPUT_LRS, NN_NUM_HIDDEN_LRS)

episode_count = 0
for i in range(NUM_EPISODES):

	print "Episode %d" % episode_count

	# Evaluate Rover team, assess performance
	execute_episode()

	roverDomain.select()

	roverDomain.mutateNNs(PERTURBATION)

	# Increment episode counter
	episode_count += 1
