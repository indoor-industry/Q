import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0.001, 0.1, 100)

#q-metric dispersion relation
def g(t):
    dim = 4
    L_0 = 1
    gl = t
    xi = (L_0/gl)**2
    T_squared = 1 + xi
    #g = (((dim-1)/gl)*(1-T_squared**(-2))-dim*T_squared**(-1)*(L_0**2/gl**3))
    g = -dim*T_squared**(-1)*(L_0**2/gl**3)

    return g

g_values = [np.real(g(time)) for time in t]

plt.plot(t, g_values)
plt.show()