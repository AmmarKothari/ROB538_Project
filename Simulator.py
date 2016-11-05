__author__ = "Ovunc Tuzel"

import random, math, utils
import numpy as np
from NN_Unsupervised import NeuralNet
from operator import attrgetter
import copy

# =======================================================
# Simulator
# =======================================================
class Simulator(object):

	# Constructor
	def __init__(self,
		min_sensor_dist,
		max_sensor_dist,
		num_rovers,
		num_pois,
		world_width,
		world_height):

		self.min_sensor_dist	= min_sensor_dist
		self.max_sensor_dist	= max_sensor_dist
		self.num_rovers			= num_rovers
		self.num_pois			= num_pois
		self.world_width		= world_width
		self.world_height		= world_height
		self.rover_list			= []
		self.poi_list			= []


	# Initialize world
	def init_world(self, poi_min_vel, poi_max_vel, holonomic):

		# Initializing rovers
		for i in range(self.num_rovers):
			self.add_rover(random.randint(0, self.world_height), random.randint(0, self.world_width), 0, holonomic)

		# Initializing POIs
		for i in range(self.num_pois):
			self.add_poi(random.randint(0, self.world_height), random.randint(0, self.world_width), random.randint(0, 360))

		# Setting POIs initial velocities
		for poi in self.poi_list:
			poi.heading = random.randint(0, 360)
			poi.set_vel_lin(random.uniform(poi_min_vel, poi_max_vel))

	def init_world_custom(self, poi_min_vel, poi_max_vel, poi_locations, rover_locations):

		# Initializing rovers
		for i in range(self.num_rovers):
			self.add_rover(rover_locations[i][0], rover_locations[i][1], rover_locations[i][2])

		# Initializing POIs
		for i in range(self.num_pois):
			self.add_poi(poi_locations[i][0], poi_locations[i][1], poi_locations[i][2])

		# Setting POIs initial velocities
		for poi in self.poi_list:
			poi.set_vel_lin((poi_max_vel + poi_min_vel)/2.0)

	# Reset performance counter
	def reset_performance(self, pop_set):
		for rover in self.rover_list:
			rover.population[pop_set].performance = 0

	# Reset performance counter
	def get_performance(self, roverNum):
		performance = []
		for nn in self.rover_list[roverNum].population:
			performance.append(nn.performance)
		return performance

	# Loading NN weights
	def load_bestWeights(self, filename):
		for i in range(self.num_rovers):
			for nn in self.rover_list[i].population:
				nn.load_weights(filename+"R"+str(i)+"_")

	# Storing NN weights
	def store_bestWeights(self, filename):
		for i in range(self.num_rovers):
			self.rover_list[i].population[-1].store_weights(filename+"R"+str(i)+"_")

	# Iterate the world simulation
	def sim_step(self, pop_set):

		# POIs step
		for poi in self.poi_list:
			poi.sim_step(self.world_width,self.world_height)

		# Rovers step
		for rover in self.rover_list:
			inputs = self.return_NN_inputs(rover)
			outputs = rover.population[pop_set].forward(inputs)
			rover.sim_step(outputs)

		# Compute rover observation values of POIs
		for poi in self.poi_list:
			for i in range(len(self.rover_list)):
				poi.obs[i] = utils.cap_distance(poi.pos, self.rover_list[i].pos, self.min_sensor_dist)

	# =======================================================
	# Rewards
	# =======================================================

	# Local reward
	def local_reward(self, pop_set):
		for poi in self.poi_list:
			self.rover_list[np.argmax(poi.obs)].population[pop_set].performance += np.max(poi.obs)

	# Global reward computation
	def compute_global_reward(self, pop_set, excluded_rover=-1):
		reward = 0
		for poi in self.poi_list:
			
			# Observations to this POI
			aux = poi.obs

			# Eliminating excluded_rover observations
			if excluded_rover > -1:
				aux[excluded_rover] = 0

			# Getting the max reward
			reward += np.max(aux)

		return reward

	# Global reward
	def global_reward(self, pop_set):

		global_rwd = self.compute_global_reward(pop_set)

		# Assigning global reward
		for rover in self.rover_list:
			rover.population[pop_set].performance = global_rwd

	# Differential reward
	def diff_reward(self, pop_set):

		global_rwd = self.compute_global_reward(pop_set)

		# global_reward(pop_set) - 
		for i in range(len(self.rover_list)):
			self.rover_list[i].population[pop_set].performance = global_rwd - self.compute_global_reward(pop_set, i)

	# =======================================================
	# =======================================================

	# Initialize NNs for each rover
	def initRoverNNs(self, pop_size, inputLayers, outputLayers, hiddenLayers, input_scaling, output_scaling):
		for rover in self.rover_list:
			for i in range(pop_size):
				rover.population.append(NeuralNet(inputLayers, outputLayers, hiddenLayers, input_scaling, output_scaling))


	# Initialize NNs for each rover
	def mutateNNs(self, mutation_std):
		for rover in self.rover_list:
			mutantlist = []
			for nn in rover.population:
				mutant = copy.deepcopy(nn)
				mutant.perturb_weights(mutation_std)
				mutantlist.append(mutant)
			rover.population += mutantlist

	# Selecting best NNs
	def select(self):

		# Retain k best
		for rover in self.rover_list:
			rover.population = self.remove_worst(rover.population)

	# Removing the worst performing rover
	def remove_worst(self, nn_list, k=None):
		if k == None:
			k = len(nn_list)/2
		nn_list = sorted(nn_list, key=attrgetter('performance'))
		del nn_list[:k]
		return nn_list

	# Registering new POI
	def add_poi(self, x=0, y=0, heading=0, value=1.0):
		self.poi_list.append(Poi(x, y, heading, value, self.num_rovers))


	# Registering new rover
	def add_rover(self, x=0, y=0, heading=0, holonomic=1):
		self.rover_list.append(Rover(x, y, heading, holonomic))


	# Resetting agents to random or initial starting position
	def reset_agents(self, rnd_pois = 1, rnd_rover_pos = 1):

		# Resetting POIs
		for poi in self.poi_list:
			poi.pos			= poi.init_pos
			poi.heading		= poi.init_head
			if rnd_pois:
				poi.pos		= random.randint(0,self.world_width), random.randint(0,self.world_height)
				poi.heading	= random.randint(0,360)

		# Resetting Rovers
		for rover in self.rover_list:
			rover.pos		= rover.init_pos
			rover.heading 	= rover.init_head
			if rnd_rover_pos:
				rover.pos = random.randint(0,self.world_width), random.randint(0,self.world_height)
				if not rover.holonomic:
					rover.heading = random.randint(0, 360)

	# Computing sensor measurement
	def measure_sensor(self, agentList, quadrant, rover):
		sum = 0
		for agent in agentList:
			if agent != rover:
				angle = utils.get_angle(utils.vect_sub(agent.pos, rover.pos))
				relative_angle = (angle - rover.heading) % (2*math.pi)
				if utils.check_quadrant(relative_angle, quadrant):
					sum += agent.value*utils.cap_distance(agent.pos, rover.pos, self.min_sensor_dist)
		return sum
	
	def return_POI_vel(self, poiList, max_dist = 500):
                min_dist = 10
                sum = np.zeros([4,2])
                count = np.zeros(4)
                for poi in poiList:
                	rover_pos = rover.pos
                    	# get quadrant of POI
                	if math.isnan(rover.pos[0]):
                		rover_pos = [0,0]
                	vect = utils.vect_sub(poi.pos, rover_pos)
                	dist = max(utils.get_norm(vect), MIN_SENSOR_DIST)
                	dist_2 = dist ** 2
                    angle = utils.get_angle(vect) % (2*math.pi ) # Between 0 to 2pi
                    relative_angle = (angle - self.heading + math.pi/2) % (2*math.pi)
                    q = utils.get_quadrant(relative_angle) - 1
                    
                    #get relative velocity of POI to agent.
                    poi_vel_vect = np.array(poi.vel_lin)
                    rover_vel_vect = np.array(rover.vel_lin)
                    rel_vel_vect = rover_vel_vect - poi_vel_vect
                    if rover.pos[0] < poi.pos[0]:
				rel_vel_vect[0] *= -1
                    if rover.pos[1] < poi.pos[1]:
                		rel_vel_vect[1] *= -1

                    # update average velocity vector
                    count[q] += 1
                    sum[q][0] = (sum[q][0] * (count[q] - 1) + rel_vel_vect[0]/dist_2) / count[q]
                    sum[q][1] = (sum[q][1] * (count[q] - 1) + rel_vel_vect[1]/dist_2) / count[q]

                return sum
	
	# Gathering all sensor measurements
	def return_NN_inputs(self, rover):

		inputs = []

		# Sensing rovers
		for i in range(4):
			inputs.append(self.measure_sensor(self.rover_list, i, rover))

		# Sensing POIs
		for i in range(4):
			inputs.append(self.measure_sensor(self.poi_list, i, rover))

		# print inputs
		return inputs
# =======================================================
# =======================================================



# =======================================================
# General agent that models POIs and rovers
# =======================================================
class Agent(object):

	# Constructor
	def __init__(self, posx, posy, heading, value = 1.0):
		self.init_pos	= (posx, posy)		# Starting position
		self.pos		= (posx, posy)		# Current position
		self.vel_lin	= (0.0, 0.0)		# Linear velocity
		self.vel_ang	= 0.0				# Angular velocity (rad/sec)
		self.init_head  = heading  			# Starting position
		self.heading	= heading			# Heading direction (rad)	
		self.value		= value				# Utility value

	# Update heading using angular velocity
	def update_heading(self):
		self.heading += self.vel_ang
		self.set_vel_lin(self.get_vel_lin())

	# Update position using linear velocity
	def update_pos(self):
		self.pos = utils.vect_sum(self.pos, self.vel_lin)

	# Set heading and update velocity accordingly
	def set_heading(self, heading):
		self.heading = heading
		self.set_vel_lin(self.get_vel_lin())

	# Get absolute velocity
	def get_vel_lin(self):
		return utils.get_norm(self.vel_lin)

	# Set absolute velocity
	def set_vel_lin(self,vel_lin_abs):
		self.vel_lin = vel_lin_abs*math.cos(self.heading), vel_lin_abs*math.sin(self.heading)

	# Wall bouncing
	def bounce_walls(self, world_width, world_height):

		# Check left-right wall collisions
		if self.pos[0] > world_width or self.pos[0] < 0:
			self.set_heading((1*math.pi - self.heading) % (math.pi * 2))

		# Check top-bottom wall collisions
		if self.pos[1] > world_height or self.pos[1] < 0:
			self.set_heading((2*math.pi - self.heading) % (math.pi * 2))
# =======================================================
# =======================================================



# =======================================================
# POI agent
# =======================================================
class Poi(Agent):
	def __init__(self, posx, posy, heading, value, num_rovers):
		Agent.__init__(self, posx, posy, heading, value)
		self.obs = np.zeros(num_rovers)

	# Simulation step for the POIs
	def sim_step(self, world_width, world_height):
		self.update_heading();
		self.update_pos();
		self.bounce_walls(world_width, world_height)
# =======================================================
# =======================================================



# =======================================================
# Rover agent
# =======================================================
class Rover(Agent):

	def __init__(self, posx, posy, heading, holonomic):
		Agent.__init__(self, posx, posy, heading, 1.0)
		self.population = []
		self.holonomic = holonomic

	# Simulation step for the rovers
	def sim_step(self, nn_outputs):

		# print self.vel_ang, utils.get_norm(self.vel_lin)
		if self.holonomic:
			self.vel_lin = (nn_outputs[0],nn_outputs[1])
			self.update_pos();
		else:
			self.vel_ang = nn_outputs[0]
			self.set_vel_lin(nn_outputs[1])
			self.update_heading();
			self.update_pos();
# =======================================================
# =======================================================
