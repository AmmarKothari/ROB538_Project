__author__ = "Ovunc Tuzel"

import random, math, utils

class RoverDomain(object):

	# Constructor
	def __init__(self, height=240, width=240):
		self.height = height
		self.width = width
		self.rover_list = []
		self.poi_list = []

	def add_poi(self, x=0, y=0, heading=0, value=1.0):
		poi = Poi(x, y, heading, value)
		poi.roverDomain = self
		self.poi_list.append(poi)

	def add_rover(self, x=0, y=0, heading=0):
		rover = Rover(x, y, heading)
		rover.roverDomain = self
		self.rover_list.append(rover)

	# Resetting agents to random or initial starting position
	def reset_agents(self, rnd_pois = 1, rnd_rovers = 1):

		# Resetting POIs
		for i in self.poi_list:
			i.pos = i.init_pos if not rnd_pois else (random.randint(0,self.width), random.randint(0,self.height))

		# Resetting Rovers
		for i in self.rover_list:
			i.pos = i.init_pos if not rnd_pois else (random.randint(0,self.width), random.randint(0,self.height))

class Agent(object):

	# Constructor
	def __init__(self, posx, posy, heading):
		self.init_pos	= (posx, posy)		# Starting position
		self.pos		= (posx, posy)		# Current position
		self.vel_lin	= (0.0,0.0)			# Linear velocity
		self.vel_ang	= 0.0				# Angular velocity (rad/sec)
		self.heading	= heading			# Heading direction (rad)
		self.roverDomain = None				

	# Simulation step
	def sim_step(self):
		self.update_heading();
		self.update_pos();
		self.bounce_walls()

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
	def bounce_walls(self):

		# Check left-right wall collisions
		if self.pos[0] > self.roverDomain.width or self.pos[0] < 0:
			self.set_heading((1*math.pi - self.heading) % (math.pi * 2))

		# Check top-bottom wall collisions
		if self.pos[1] > self.roverDomain.height or self.pos[1] < 0:
			self.set_heading((2*math.pi - self.heading) % (math.pi * 2))

# POI agent
class Poi(Agent):
	def __init__(self, posx, posy, heading, value):
		Agent.__init__(self, posx, posy, heading)
		self.value = value

# Rover agent
class Rover(Agent):
	def __init__(self, posx, posy, heading):
		Agent.__init__(self, posx, posy, heading)
		self.population = []

	def return_sensor_poi(self, poiList, quadrant, max_dist=500):
		min_dist = 10
		sum = 0
		for poi in poiList:
			vect = utils.vect_sub(poi.pos, self.pos)
			dist = utils.get_norm(vect)
			angle = utils.get_angle(vect) % (2*math.pi ) # Between 0 to 2pi
			relative_angle = (angle - self.heading + math.pi/2) % (2*math.pi)
			# print (angle + self.heading + math.pi/2)
			# print 'Vect: ', vect
			# print 'Angle: ', angle*360/2/math.pi, relative_angle*360/2/math.pi
			if dist < max_dist and utils.check_quadrant(relative_angle, quadrant):
				# print 'I SEE YOU', quadrant
				sum += poi.value / max(dist**2, min_dist**2)
		return sum

	def return_sensor_rover(self, roverList, quadrant, max_dist=500):
		min_dist = 10
		sum = 0
		for rover in roverList:
			vect = utils.vect_sub(rover.pos, self.pos)
			dist = utils.get_norm(vect)
			angle = utils.get_angle(vect) % (2 * math.pi)  # Between 0 to 2pi
			relative_angle = (angle - self.heading + math.pi / 2) % (2 * math.pi)
			# print 'Vect: ', vect
			# print 'Angle: ', angle*360/2/math.pi, relative_angle*360/2/math.pi
			if dist < max_dist and utils.check_quadrant(relative_angle, quadrant):
				# print 'I SEE YOU', quadrant
				sum += 1 / max(dist ** 2, min_dist ** 2)
		return sum

	def return_NN_inputs(self):
		inputs = []
		for i in range(4):
			inputs.append(self.return_sensor_rover(self.roverDomain.rover_list, i))
		for i in range(4):
			inputs.append(self.return_sensor_poi(self.roverDomain.poi_list, i))
		return inputs
