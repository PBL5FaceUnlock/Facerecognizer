from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class Converter:
    def __init__(self):
        """ CHANGE THE SAMPLE DATA HERE """
        self.distance_samples = np.array([0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.2], dtype=float)
        self.similarity_samples = np.array([100, 95, 90, 85, 82, 82, 75, 50, 40, 20,10 ], dtype=float)

        self.tck = interpolate.splrep(self.distance_samples, self.similarity_samples)
        self.xnew = np.linspace(0.58, 1.2, num=101, endpoint=True)
        self.ynew = interpolate.splev(self.xnew, self.tck, der=0)

    def convert_dis2sim(self, distance):
        index = np.searchsorted(self.xnew, distance)
        if index < len(self.ynew):
            sim = int(self.ynew[index])
        else:
            sim = self.ynew[-1]

        if sim > 100:
            sim = 100
        elif sim < 0:
            sim = 0
        return sim

    def show_figure(self):

        plt.plot(self.distance_samples, self.similarity_samples, "o", self.xnew, self.ynew)
        plt.legend(["data", "spline-cubic"], loc="best")
        plt.show()


#dis = Converter()
# dis.show_figure()

#print("Similarity:", dis.convert_dis2sim(0.9))


