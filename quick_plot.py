import matplotlib.pyplot as plt
import pickle
import numpy as np
import pdb


FILE_NAME = 'RS_2016-11-03 17:22:24.366048_POI10_ROV1'

# FILE_NAME = 'Performance_10POI'





FILE_NAME += '.pickle'


# Getting back the objects:
with open(FILE_NAME) as f:  # Python 3: open(..., 'rb')
    best_perf, avg_perf, std_perf = pickle.load(f)


trunc = int(len(best_perf) % 10)
index = np.nonzero(best_perf)
best_perf_arr = np.array(best_perf[index])
plt.plot(best_perf_arr, 'ro-')
plt.show()
# best_perf_arr = np.reshape(best_perf_arr, [10,-1])
# best_perf_mean = np.mean(best_perf_arr,1)
