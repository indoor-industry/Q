import numpy as np
import matplotlib.pyplot as plt

extent = 50
samples = 1000
m = 100
L_0 = 1
#gl = np.sqrt(np.abs(t**2))#-x**2))
gl = 1
k = np.linspace(-extent, extent, samples)
def dr(k):
    re = k**2 + m**2
    mod = re**2 + m**2*(L_0**4/(gl**6))*(3*L_0**2/(gl**2)+2)**2
    return (0.5*(mod+re))**0.5 + 1j*(0.5*(mod-re))**0.5

plt.plot(k, np.real(dr(k)))
plt.plot(k)
plt.show()