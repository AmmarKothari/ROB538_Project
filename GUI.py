from Tkinter import *
from RoverDomain import *
from NN_Unsupervised import NeuralNet
from operator import attrgetter
import time, copy
import numpy as np

def init_canvas(roverDomain):
    """Initialize the canvas"""
    global master, w
    master = Tk()
    canvas_width = roverDomain.width
    canvas_height = roverDomain.height

    master.title("Rob538 Project - Rover Domain")
    w = Canvas(master, width=canvas_width, height=canvas_height, background='#FFFFFF')
    w.pack()

def get_points_triangle(agent, l=5):
    x = agent.pos[0]
    y = agent.pos[1]
    t = agent.heading
    p1 = [x + l * math.sin(t), y - l * math.cos(t)]
    p2 = [x - l * math.sin(t), y + l * math.cos(t)]
    p3 = [x + 3 * l * math.cos(t), y + 3 * l * math.sin(t)]
    return [p1,p2,p3]

def init_agents():
    """ Initialize agents here! """
    for i in range(3):
        roverDomain.add_rover(180 + 20*i, 200)

    for i in range(20):
        roverDomain.add_poi(random.randint(0, 350), random.randint(0, 470))

    for poi in roverDomain.poi_list:
        poi.speed = random.randint(2, 8) / 20.0
        poi.heading = random.randint(0, 360) / 360.0 * 2 * math.pi

def draw_agents(roverDomain):
    global objList
    clear_objList()
    objList = []
    for agent in roverDomain.rover_list:
        obj = w.create_polygon(get_points_triangle(agent), fill='orange')
        objList.append(obj)
    for poi in roverDomain.poi_list:
        obj = w.create_polygon(get_points_triangle(poi, l=3), fill='red')
        objList.append(obj)
    w.update()

def clear_objList():
    global objList
    for obj in objList:
        w.delete(obj)

def evaluate_rover_team(team, disp=True):
    """Takes a team of networks to evaluate rovers, each network is given a performance value at the end"""
    avgiter = 1
    rewardsum = 0
    for n in range(avgiter):
        roverDomain.reset_agents(opts='RandomPR')
        for i in range(simsteps):
            for poi in roverDomain.poi_list:
                poi.walk()

            NN_index = 0
            for rover in roverDomain.rover_list:
                NN = team[NN_index]
                inputs = rover.return_NN_inputs()
                outputs = NN.forward(inputs)
                rover.walk(outputs)
                NN_index += 1

            if disp:
                draw_agents(roverDomain)
        #rewardsum += calculate_reward(catch_time, steps) # REWARD STRUCTURES COME HERE!

    #NN.performance = rewardsum / avgiter

def pick_network_egreedy(list, epsilon=0.9):
    """" UNUSED - With probabilty epsilon, return best network in the list. Else return random. """
    random.shuffle(list)
    d = random.randint(1,100)
    if d < epsilon*100:
        return max(list, key=attrgetter('performance'))
    else:
        return random.choice(list)

def return_worst(list):
    """ UNUSED """
    random.shuffle(list)
    worst = min(list, key=attrgetter('performance'))
    return worst

def remove_worst(NNlist, k=None):
    if k == None:
        k = len(NNlist)/2
    for i in range(k):
        random.shuffle(NNlist)
        worst = min(NNlist, key=attrgetter('performance'))
        NNlist.remove(worst)
    return NNlist

global objList
objList = []
roverDomain = RoverDomain(360,480)
init_canvas(roverDomain)
init_agents()

population = []
population_size = 10
perturbation = 0.25
simsteps = 1000
generation_count = 0

# INITIALIZE <population_size> Neural Nets with 8 input, 2 output and 10 hidden units for every rover.
for rover in roverDomain.rover_list:
    for i in range(population_size):
        NN = NeuralNet(8, 2, 10)
        rover.population.append(NN)

# EVOLUTION ================================================
iterations = 10
for i in range(iterations):
    # Mutate each NN to have 2k NNs
    for rover in roverDomain.rover_list:
        mutantlist = []
        for NN in rover.population:
            mutant = copy.deepcopy(NN)
            mutant.perturb_weights(perturbation)
            mutantlist.append(mutant)
        rover.population += mutantlist

    # Randomly select one from each to form a team
    team = []
    for rover in roverDomain.rover_list:
        team.append(random.choice(rover.population)) # Same random!!!!
    # Evaluate Rover team, assess performance
    evaluate_rover_team(team)
    # Retain k best
    for rover in roverDomain.rover_list:
        rover.population = remove_worst(rover.population)
    # Increment Gen counter
    generation_count += 1
# ==========================================================


# iterations = 10000
# for i in range(iterations):
#     for rover in roverDomain.rover_list:
#         # PICK network using e-greedy
#         to_mutate = pick_network_egreedy(rover.population)
#         # MUTATE selected network
#         mutant = copy.deepcopy(to_mutate)
#         mutant.perturb_weights(perturbation)
#         # USE and EVALUATE this network
#
#         evaluate_rovers(mutant)
#         # REINSERT mutant
#         rover.population.append(mutant)
#         # REMOVE worst network
#         worst = return_worst(rover.population)
#         rover.population.remove(worst)
#
#     generation_count += 1
#     # time.sleep(0.01)

mainloop()

