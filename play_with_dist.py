from scipy import stats
import matplotlib.pyplot as pyplot
import numpy as np

loc = 0.6888828116488848
scale = 0.17928022000684693

pyplot.figure()

rv = stats.logistic(-1, 0.5)
x = np.linspace(-2, 2, 200)
pyplot.plot(x, rv.pdf(x))
pyplot.show()
