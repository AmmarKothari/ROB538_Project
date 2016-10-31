import math, utils

angle = utils.get_angle((-90, 193)) % (2 * math.pi)  # Between 0 to 2pi
print 'Angle: ', angle*180/math.pi

# print math.atan2(1,0)