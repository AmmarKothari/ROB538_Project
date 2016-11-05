import matplotlib.pyplot as plt
import pickle
import numpy as np
import pdb
import os
import Tkinter, tkFileDialog
import csv



##### PARAMETERS  ######

POINTS_GRAPH = 10


##########################

# FILE_NAME += '.pickle'
directory = os.getcwd()
files = os.listdir(directory)
# Ask user to select files
root = Tkinter.Tk()
filename = tkFileDialog.askopenfiles(mode='r',
									parent=root,
									initialdir=".",
                                    title='Please select file(s) to plot')

print(filename)

output_vals = list()
max_vals = list()
names = list()
count = 0
for f in filename:
	output_vals.append([])
	names.append(f.name)
	reader = csv.reader(f, delimiter=',')
	for val in reader:
		output_vals[count].append(val)

	count += 1

#convert to numpy array
output_vals = np.array(output_vals).astype(np.float)

# sort values in each generation from largest to smallest
vals_sort = np.sort(output_vals[:][:][:])
vals_max = np.amax(output_vals[:][:][:], axis = 2)



trunc = len(vals_max[0]) % 10
average_vals = list()
std_vals = list()
for i in range(len(vals_max)):
	reshaped_vals = np.reshape(vals_max[i][:-trunc],(-1,10))
	average_vals.append(np.mean(reshaped_vals, axis = 0))
	std_vals.append(np.std(reshaped_vals, axis = 0))
	plt.errorbar(np.arange(POINTS_GRAPH), average_vals[i], yerr = std_vals[i], label = names[i].split('/')[-1])


plt.xlabel = 'Reward'
plt.ylabel = str(len(reshaped_vals)) + 's of Iterations'
plt.legend()

plt.show()
# print(output_vals)


# for f_current in F:
# 	print(f_current)
# 	# Getting back the objects:
# 	with open(f_current) as f:  # Python 3: open(..., 'rb')
# 		try:
# 			best_perf, avg_perf, std_perf, InputParameters = pickle.load(f)
# 		except:
# 			print('No input parameters')
# 			best_perf, avg_perf, std_perf = pickle.load(f) 

# 	# pdb.set_trace()

# 	index = np.nonzero(best_perf[0])[0]
# 	trunc = int(len(index) % 10)
# 	if trunc == 0:
# 		best_perf_arr = np.array(best_perf[0][index])
# 	else:
# 		best_perf_arr = np.array(best_perf[0][index[:-trunc]])
# 	best_perf_arr = np.reshape(best_perf_arr, [10,-1])

# 	x = np.arange(10) + 0.5
# 	best_perf_mean = np.mean(best_perf_arr,1)
# 	best_perf_std = np.std(best_perf_arr,1)
# 	# plt.plot(best_perf_mean, 'ro-')
# 	plt.errorbar(x, best_perf_mean, yerr = best_perf_std, color='blue', marker='o', linestyle=':')
# 	plt.xlabel(str(len(best_perf_arr[0])) + 's of Generations')
# 	plt.ylabel('Reward')
# 	POI_Count = InputParameters['NUM_POIS']
# 	ROVER_COUNT =  InputParameters['NUM_ROVERS']
# 	plt.title('POIs: %s, Rovers: %s' %(POI_Count, ROVER_COUNT))
# 	# pdb.set_trace()
# 	plt.show()
