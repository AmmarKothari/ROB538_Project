import matplotlib.pyplot as plt
import pickle
import numpy as np
import pdb
import os



Plot_All = 0



FILE_NAME = 'RS_2016-11-03 17:22:24.366048_POI10_ROV1'

# FILE_NAME = 'Performance_10POI'

FILE_NAME = 'RS_2016-11-03 20:58:47.604165_POI10_ROV1'
FILE_NAME = 'RS_2016-11-03 23:37:49.103581_POI10_ROV1'
FILE_NAME = 'RS_2016-11-04 01:04:46.443121_POI10_ROV1'
FILE_NAME = 'RS_2016-11-04 01:51:24.843199_POI10_ROV1'
FILE_NAME = 'RS_2016-11-04 02:02:52.550221_POI10_ROV1'

FILE_NAME += '.pickle'

directory = os.getcwd()
files = os.listdir(directory)
F_names = []
for f in files:
		if '.pickle' in f:
			F_names.append(f)


if Plot_All:
	F = F_names
else:
	F = [FILE_NAME]

for f_current in F:
	print(f_current)
	# Getting back the objects:
	with open(f_current) as f:  # Python 3: open(..., 'rb')
		try:
			best_perf, avg_perf, std_perf, InputParameters = pickle.load(f)
		except:
			print('No input parameters')
			best_perf, avg_perf, std_perf = pickle.load(f) 

	# pdb.set_trace()

	index = np.nonzero(best_perf[0])[0]
	trunc = int(len(index) % 10)
	if trunc == 0:
		best_perf_arr = np.array(best_perf[0][index])
	else:
		best_perf_arr = np.array(best_perf[0][index[:-trunc]])
	best_perf_arr = np.reshape(best_perf_arr, [10,-1])

	x = np.arange(10) + 0.5
	best_perf_mean = np.mean(best_perf_arr,1)
	best_perf_std = np.std(best_perf_arr,1)
	# plt.plot(best_perf_mean, 'ro-')
	plt.errorbar(x, best_perf_mean, yerr = best_perf_std, color='blue', marker='o', linestyle=':')
	plt.xlabel(str(len(best_perf_arr[0])) + 's of Generations')
	plt.ylabel('Reward')
	POI_Count = InputParameters['NUM_POIS']
	ROVER_COUNT =  InputParameters['NUM_ROVERS']
	plt.title('POIs: %s, Rovers: %s' %(POI_Count, ROVER_COUNT))
	# pdb.set_trace()
	plt.show()
