import math, utils

angle = utils.get_angle((-90, 193)) % (2 * math.pi)  # Between 0 to 2pi
print 'Angle: ', angle*180/math.pi
print [i for i in range(4)]
print 13/2