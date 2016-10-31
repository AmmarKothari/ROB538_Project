import math

def get_norm(vect):
    return (vect[0] ** 2 + vect[1] ** 2) ** 0.5

def get_angle(vect):
    return math.atan2(vect[1], vect[0])

def vect_sub(vect1, vect2):
    return (vect1[0] - vect2[0], vect1[1] - vect2[1])

def check_quadrant(angle, quadrant):
    pi = math.pi
    if quadrant == 1 and (7*pi/4 < angle < 2*pi or 0 < angle < pi/4):
        return True
    elif (quadrant-1)*pi/2 - pi/4 < angle < (quadrant-1)*pi/2 + pi/4:
        return True
    else:
        return False