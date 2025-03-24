import numpy as np
import matplotlib.pyplot as plt

data = np.load("./score_data/cs_mri_data.npz")
h_all = data['h']

h = h_all[0]
abs_h = np.abs(h).flatten()
abs_h_sorted = np.sort(abs_h)[::-1]

plt.plot(abs_h_sorted)
plt.yscale("log")
plt.title("values |h|")
plt.xlabel("sorted indexes")
plt.ylabel("amplitude (log)")
plt.grid(True)
plt.show()
