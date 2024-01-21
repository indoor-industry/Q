import numpy as np
import matplotlib.pyplot as plt

m = 100
d=2
t = np.linspace(0.1, 2.5, 100)
omega = np.linspace(0.1, 0.5, 5)

#dispersion relation
#omega = k**2/(2*m)
#omega = np.sqrt(k**2 + m**2)

def delta_S(t):
    return ((2*np.pi)/(m*d))*t

def delta_KG(t, omega):
    return ((2*np.pi)/d)*(t/omega)

def delta_Q_real(t, omega):

    #q-metric dispersion relation
    dim = 4
    L_0 = 1
    gl = t
    xi = (L_0/gl)**2
    T_squared = 1 + xi
    g = (((dim-1)/gl)*(1-T_squared**(-2))-dim*T_squared**(-1)*(L_0**2/gl**3))
    
    #omega = 0.5*(-1j*g + np.emath.sqrt(-g**2 + 4*T_squared**(-1)*(T_squared**(-1)*k**2 + m**2)))

    return ((2*np.pi)/d)*(t/(omega**2+g**2/4))*omega

fig, axs = plt.subplots(3)
fig.suptitle('Fringe maxima separations')
axs[0].set_title('Schroedinger')
axs[1].set_title('KG')
axs[2].set_title('Q')

delta_S_values = [delta_S(time) for time in t]
axs[0].plot(t, delta_S_values)

for omegas in omega:
    delta_KG_values = [delta_KG(time, omegas) for time in t]
    axs[1].plot(t, delta_KG_values, label='omega={}'.format(omegas))

    delta_Q_values = [delta_Q_real(time, omegas) for time in t]
    axs[2].plot(t, delta_Q_values)

axs[1].legend()
plt.show()