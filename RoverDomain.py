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
        self.heading = linear

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
        self.speed = 0
        self.heading = heading
        self.roverDomain = None
        self.population = []

    def setSpeed(self, linear, angular):
        self.speed = linear
        self.heading = linear

    def walk(self, command):
        self.pos[0] += math.cos(self.heading) * self.speed
        self.pos[1] += math.sin(self.heading) * self.speed

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

def reset_agents(opts = 'set'):
    dimx = len(gridmatrix[0]) - 1
    dimy = len(gridmatrix) - 1
    if opts == 'set':
        for i in prd_list:
            i.pos = i.startpos
        for i in bait_list:
            i.pos = i.startpos
    elif opts == 'randompr':
        for i in prd_list:
            i.pos = (random.randint(0,dimx), random.randint(0,dimy))
        for i in bait_list:
            i.pos = i.startpos
    elif opts == 'randombait':
        for i in prd_list:
            i.pos = i.startpos
        for i in bait_list:
            i.pos = (random.randint(0,dimx), random.randint(0,dimy))
    elif opts == 'randomall':
        for i in prd_list:
            i.pos = (random.randint(0,dimx), random.randint(0,dimy))
        for i in bait_list:
            i.pos = (random.randint(0,dimx), random.randint(0,dimy))
    for i in prd_list:
        i.startpos = i.pos
    for i in bait_list:
        i.startpos = i.pos

    update_gridmatrix()

def reset_random():
    for i in prd_list:
        i.pos = i.startpos
    for i in bait_list:
        i.pos = (random.randint(0, len(gridmatrix[0])-1), random.randint(0, len(gridmatrix)-1))

    update_gridmatrix()
