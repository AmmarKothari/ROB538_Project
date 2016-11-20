import math

def cap_distance(vect1, vect2, min_dist_sqr):
    dist_sqr = max(get_norm_sqr(vect_sub(vect1, vect2)), min_dist_sqr)
    return 1.0 / dist_sqr

def get_norm_sqr(vect):
    return (vect[0] ** 2 + vect[1] ** 2)

def get_norm(vect):
    return (vect[0] ** 2 + vect[1] ** 2) ** 0.5

def get_angle(vect):
    return math.atan2(vect[1], vect[0]) % (2 * math.pi) 

def vect_sub(vect1, vect2):
    return (vect1[0] - vect2[0], vect1[1] - vect2[1])

def vect_sum(vect1, vect2):
    return (vect1[0] + vect2[0], vect1[1] + vect2[1])

def check_quadrant(angle, quadrant):
    pi = math.pi
    if quadrant == 0 and (7*pi/4 < angle < 2*pi or 0 < angle < pi/4):
        return True
    elif quadrant*pi/2 - pi/4 < angle < quadrant*pi/2 + pi/4:
        return True
    else:
        return False


def get_quadrant(angle):
    step = math.pi/4

    quadrants = [1, 2, 2, 3, 3, 4, 4, 1]

    i = angle // step
    if math.isnan(i):
        pdb.set_trace()
    return quadrants[int(i)]