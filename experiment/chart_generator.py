import numpy as np
import matplotlib.pyplot as plt

scores_file_name = "results.csv"
scores = np.genfromtxt(scores_file_name, delimiter=",")

x1 = [x for x in range(1, 22)]
y1 = scores[0, :]
y2 = scores[1, :]
y3 = scores[2, :]
y4 = scores[3, :]
y5 = scores[4, :]
y6 = scores[5, :]

plt.plot(x1, y1, label="kNN, k=3, Manhattan")
plt.plot(x1, y2, label="kNN, k=3, Euklidesowa")
plt.plot(x1, y3, label="kNN, k=5, Manhattan")
plt.plot(x1, y4, label="kNN, k=5, Euklidesowa")
plt.plot(x1, y5, label="kNN, k=7, Manhattan")
plt.plot(x1, y6, label="kNN, k=7, Euklidesowa")

plt.title("Dokładność algorytmu w zależności od liczby cech")
plt.ylabel("Dokładność")
plt.xlabel("Liczba cech")
plt.legend()
plt.savefig(fname='../images/results_chart.eps', orientation='portrait')
plt.show()
