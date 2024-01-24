import numpy as np
import matplotlib.pyplot as plt

m = 1
d = 1
t = np.linspace(0.1, 3, 1000)
k = np.linspace(0.1, 5, 5)

def delta_S(t):
    v_g = k/m
    omega = k**2/(2*m)
    return ((2*np.pi)/(m*d))*t

def delta_KG(t, k):
    omega = np.sqrt(k**2 + m**2)
    v_g = k/omega
    return ((2*np.pi)/d)*(t/omega)

def delta_Q(t, k):
    #q-metric dispersion relation
    dim = 4
    L_0 = 1
    gl = t
    xi = (L_0/gl)**2
    T_squared = 1 + xi
    g = -dim*T_squared**(-1)*(L_0**2/gl**3)
    squareroot = np.emath.sqrt(-g**2 + 4*T_squared**(-1)*(T_squared**(-1)*k**2 + m**2))
    v_g = (2*T_squared**(-2)*k)/squareroot
    return ((4*np.pi)/d)*((T_squared**(-2))/(squareroot))*t

def delta_Q_lowk(t, k):
    dim = 4
    L_0 = 1
    gl = t
    xi = (L_0/gl)**2
    T_squared = 1 + xi
    g = -dim*T_squared**(-1)*(L_0**2/gl**3)
    squareroot = np.emath.sqrt(-g**2 + 4*T_squared**(-1)*m**2)
    return (4*np.pi*T_squared**(-2)*t)/(d*squareroot)

fig, axs = plt.subplots(3)
fig.suptitle('Fringe maxima separations')
axs[0].set_title('Schroedinger')
axs[1].set_title('KG')
axs[2].set_title('Q')

delta_S_values = [delta_S(time) for time in t]
axs[0].plot(t, delta_S_values)

for ks in k:
    delta_KG_values = [delta_KG(time, ks) for time in t]
    axs[1].plot(t, delta_KG_values, label='k=%.2f'%ks)

    delta_Q_values = [delta_Q_lowk(time, ks) for time in t]
    axs[2].plot(t, np.real(delta_Q_values))

axs[1].legend()
plt.show()