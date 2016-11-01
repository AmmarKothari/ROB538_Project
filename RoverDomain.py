__author__ = "Ovunc Tuzel"

import random, math, utils

class RoverDomain(object):
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

    def reset_agents(self, opts = 'Set'):
        if opts == 'Set':
            for i in self.rover_list:
                i.pos = i.startpos
            for i in self.poi_list:
                i.pos = i.startpos
        elif opts == 'RandomR':
            for i in self.rover_list:
                i.pos = (random.randint(0,self.width), random.randint(0,self.height))
            for i in self.poi_list:
                i.pos = i.startpos
        elif opts == 'RandomP':
            for i in self.rover_list:
                i.pos = i.startpos
            for i in self.poi_list:
                i.pos = (random.randint(0,self.width), random.randint(0,self.height))
        elif opts == 'RandomPR':
            for i in self.rover_list:
                i.pos = (random.randint(0,self.width), random.randint(0,self.height))
            for i in self.poi_list:
                i.pos = (random.randint(0,self.width), random.randint(0,self.height))
        for i in self.rover_list:
            i.startpos = i.pos
        for i in self.poi_list:
            i.startpos = i.pos

class Poi(object):
    def __init__(self, posx, posy, heading, value):
        self.pos = (posx, posy)
        self.startpos = (posx, posy)
        self.type = 'poi'
        self.speed = 0
        self.heading = heading
        self.value = value
        self.roverDomain = None

    def setSpeed(self, linear, angular):
        self.speed = linear
        self.heading = angular

    def walk(self):
        newX = self.pos[0] + math.cos(self.heading) * self.speed
        newY = self.pos[1] + math.sin(self.heading) * self.speed
        self.pos = newX, newY
        self.bounce_walls()

    def bounce_walls(self):
        # Currently pois travel in a straight line and bounce from walls. Flocking behavior might be implemented.
        bound_h = self.roverDomain.height
        bound_w = self.roverDomain.width
        if self.pos[0] > bound_w or self.pos[0] < 0:
            self.heading = (math.pi - self.heading) % (math.pi * 2)
        elif self.pos[1] > bound_h or self.pos[1] < 0:
            self.heading = (2*math.pi - self.heading) % (math.pi * 2)


class Rover(object):
    def __init__(self, posx, posy, heading):
        self.pos = (posx, posy)
        self.startpos = (posx, posy)
        self.type = 'rover'
        self.speed = 0.0
        self.heading = heading
        self.angular_vel = 0.0
        self.roverDomain = None
        self.population = []

    def setSpeed(self, linear, angular):
        self.speed = linear
        self.heading = angular

    def walk(self, command=0):
        newX = self.pos[0] + math.cos(self.heading) * self.speed
        newY = self.pos[1] + math.sin(self.heading) * self.speed
        self.pos = newX, newY
        self.heading += self.angular_vel

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




def reset_random():
    for i in prd_list:
        i.pos = i.startpos
    for i in bait_list:
        i.pos = (random.randint(0, len(gridmatrix[0])-1), random.randint(0, len(gridmatrix)-1))

    update_gridmatrix()
