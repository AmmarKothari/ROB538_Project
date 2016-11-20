__author__ = "Ovunc Tuzel"

import random, math, utils
import numpy as np
from NN_Unsupervised import NeuralNet
from operator import attrgetter
import copy
import random

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
		self.global_rwd			= 0


	# Initialize world
	def init_world(self, poi_min_vel, poi_max_vel, holonomic):

		# Initializing rovers
		for i in range(self.num_rovers):
			px = np.random.normal(loc =self.world_width/2, scale =1.0);
			py = np.random.normal(loc =self.world_height/2, scale =1.0);
			self.add_rover(px, py, 0, holonomic)

		# Initializing POIs
		for i in range(self.num_pois):
			px = self.world_width/2+random.randint(-35, 35)
			py = self.world_height/2+random.randint(-35, 35)
			self.add_poi(px,py, random.randint(0, 360))

		# Setting POIs initial velocities
		for poi in self.poi_list:
			poi.heading = random.randint(0, 360)
			poi.set_vel_lin(random.uniform(poi_min_vel, poi_max_vel))

	def init_world_custom(self, poi_min_vel, poi_max_vel, poi_init_pos, rover_init_pos):

		# Initializing rovers
		for i in range(self.num_rovers):
			# self.add_rover(0,0,0)
			self.add_rover(rover_init_pos[0][0], rover_init_pos[0][1], rover_init_pos[0][2])

		# Initializing POIs
		for i in range(self.num_pois):
			self.add_poi(poi_init_pos[i][0], poi_init_pos[i][1], poi_init_pos[i][2])

		# Setting POIs initial velocities
		for poi in self.poi_list:
			poi.set_vel_lin((poi_max_vel + poi_min_vel)/2.0)

	# Reset performance counter
	def reset_performance(self, pop_set):
		for rover in self.rover_list:
			rover.population[pop_set].performance = 0

	# Reset performance counter
	def get_performance(self, nn_list):
		performance = []
		for nn in nn_list:
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
	def sim_step(self, pop_set, steering_only, max_dist_d, sensing_vel):

		# POIs step
		for poi in self.poi_list:
			poi.sim_step(self.world_width,self.world_height)

		# Rovers step
		for rover in self.rover_list:
			inputs = self.return_NN_inputs(rover, sensing_vel)
			outputs = rover.population[pop_set].forward(inputs)
			outputs = 2*max_dist_d*(outputs-0.5)
			rover.sim_step(outputs, steering_only)

		# Compute rover observation values of POIs
		for poi in self.poi_list:
			for i in range(len(self.rover_list)):
				new_value = utils.cap_distance(poi.pos, self.rover_list[i].pos, self.min_sensor_dist)
				if new_value > poi.obs[i]:
					poi.obs[i] = new_value

	# =======================================================
	# Rewards
	# =======================================================

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

		if excluded_rover < 0:
			self.global_rwd = reward

		return reward

	# Assign local reward
	def local_reward(self, pop_set):
		for poi in self.poi_list:
			self.rover_list[np.argmax(poi.obs)].population[pop_set].performance += np.max(poi.obs)

	# Assign global reward
	def global_reward(self, pop_set):
		for rover in self.rover_list:
			rover.population[pop_set].performance = self.global_rwd

	# Assign differential reward
	def diff_reward(self, pop_set):
		for i in range(len(self.rover_list)):
			self.rover_list[i].population[pop_set].performance = self.global_rwd - self.compute_global_reward(pop_set, i)

	# =======================================================
	# =======================================================

	# Initialize NNs for each rover
	def initRoverNNs(self, pop_size, inputLayers, outputLayers, hiddenLayers, input_scaling, output_scaling):
		for rover in self.rover_list:
			for i in range(pop_size):
				rover.population.append(NeuralNet(inputLayers, outputLayers, hiddenLayers, input_scaling, output_scaling, i))

	# Printing rover NNs for debugging
	def printRoverNNs(self, title, rover):
		print "=============="
		print title+":"
		for nn in rover.population:
			print "%d :: %.6f" % (nn.id,nn.performance)
		print "=============="

	# Selecting best NNs
	def select(self,k=None):

		if k == None:
			k = len(rover.population)/2

		# Retain k best
		for rover in self.rover_list:

			# Sorting NNs
			rover.population = sorted(rover.population, key=attrgetter('performance'))

			# Getting ids of the worse NNs
			rover.worse_ids = []
			for nn in rover.population[:k]:
				rover.worse_ids.append(nn.id)

			# Deleting the worse NNs from the list
			del rover.population[:k]


	# Initialize NNs for each rover
	def gen_children(self, mutation_std, k=None):
		for rover in self.rover_list:

			# Getting k parents at random
			parents = random.sample(rover.population,k)

			for parent in parents:

				# Copy of parent
				child = copy.deepcopy(parent)

				# Mutation
				child.perturb_weights(mutation_std)

				# Getting id from deceased NN
				child.id = rover.worse_ids.pop()

				# Appending children to general population
				rover.population.append(child)

			# Sorting population according to id
			rover.population = sorted(rover.population, key=attrgetter('id'))


	# Registering new POI
	def add_poi(self, x=0, y=0, heading=0, value=1.0):
		self.poi_list.append(Poi(x, y, heading, value, self.num_rovers))


	# Registering new rover
	def add_rover(self, x=0, y=0, heading=0, holonomic=1):
		self.rover_list.append(Rover(x, y, heading, holonomic))


	# Resetting agents to random or initial starting position
	def reset_agents(self, rnd_pois = 1, rnd_rover_pos = 1, rnd_custom = 0):

		# Resetting POIs
		for poi in self.poi_list:
			poi.pos			= poi.init_pos
			poi.heading		= poi.init_head
			poi.obs			= np.zeros(self.num_rovers)
			if rnd_pois:
				px = self.world_width/2+random.randint(-35, 35)
				py = self.world_height/2+random.randint(-35, 35)
				poi.pos = px, py
				poi.heading	= random.randint(0,360)
			elif rnd_custom > 0:
				(px, py) = poi.pos
				px = np.random.normal(loc =px, scale =rnd_custom)
				py = np.random.normal(loc =py, scale =rnd_custom)
				poi.pos = px, py

		# Resetting Rovers
		for rover in self.rover_list:
			rover.pos		= rover.init_pos
			rover.heading 	= rover.init_head
			if rnd_rover_pos:
				px = np.random.normal(loc =self.world_width/2, scale =1.0);
				py = np.random.normal(loc =self.world_height/2, scale =1.0);
				rover.pos	= px, py
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

	def measure_velocity_sensor(self, poiList, rover):
		min_dist = self.min_sensor_dist
		max_dist = self.max_sensor_dist
		sum = np.zeros(4)
		for poi in poiList:

			# get quadrant of POI
			vect = utils.vect_sub(poi.pos, rover.pos)
			dist = utils.get_norm(vect)
			angle = utils.get_angle(vect) % (2 * math.pi)  # Between 0 to 2pi
			relative_angle = (angle - rover.heading + math.pi / 2) % (2 * math.pi)
			q = utils.get_quadrant(relative_angle) - 1

			# get relative velocity of POI to agent.
			rel_vel_vect = poi.vel_lin
			rel_pos_vect = utils.vect_sub(rover.pos, poi.pos)
			rel_pos_norm = utils.get_norm(rel_pos_vect)
			rel_pos_unit = [rel_pos_vect[0]/rel_pos_norm, rel_pos_vect[1]/rel_pos_norm]

			dot = np.dot(rel_pos_unit, rel_vel_vect)
			normalized_dot = poi.value * dot / rel_pos_norm**2
			sum[q] += normalized_dot

		return list(sum)
	
	# Gathering all sensor measurements
	def return_NN_inputs(self, rover, sensing_vel):

		inputs = []

		# Sensing rovers
		for i in range(4):
			inputs.append(self.measure_sensor(self.rover_list, i, rover))

		# Sensing POIs
		for i in range(4):
			inputs.append(self.measure_sensor(self.poi_list, i, rover))

		# Sensing POI velocity
		if sensing_vel:
			inputs = inputs + self.measure_velocity_sensor(self.poi_list, rover)

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
		self.worse_ids = []
		self.holonomic = holonomic

	# Simulation step for the rovers
	def sim_step(self, nn_outputs, steering_only):

		# print self.vel_ang, utils.get_norm(self.vel_lin)
		if self.holonomic:
			self.vel_lin = (nn_outputs[0],nn_outputs[1])
			self.update_pos();
		else:
			self.vel_ang = nn_outputs[0]
			if steering_only >= 0:
				self.set_vel_lin(steering_only)
			else:
				self.set_vel_lin(nn_outputs[1])
			self.update_heading();
			self.update_pos();
# =======================================================
# =======================================================
