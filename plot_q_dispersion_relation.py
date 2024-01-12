import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 5, 5)
k = np.linspace(-100, 100, 10001)

fig, (ax1, ax2) = plt.subplots(1, 2)

#q-metric dispersion relation
def KG_DR(k):
    m = 1
    omega = np.sqrt(k**2+m**2)
    return omega

KG_omega_values = [KG_DR(momentum) for momentum in k]

ax1.plot(k, KG_omega_values, label='KG')


#q-metric dispersion relation
def Q_DR(t, k):
    m = 1
    dim = 4
    L_0 = 1
    gl = t
    xi = (L_0/gl)**2
    T_squared = 1 + xi
    g = T_squared**(-1)*(((dim-1)/gl)*(T_squared-T_squared**(-1))-dim*T_squared*(L_0**2/gl**3))
    #positive solution
    omega = 0.5*(1j*g + np.sqrt(-g**2 + 4*T_squared**(-1)*(T_squared**(-1)*k**2 + m**2)))
    #negative solution
    #omega = 0.5*(1j*g - np.sqrt(-g**2 + 4*T_squared**(-1)*(T_squared**(-1)*k**2 + m**2)))

    return omega

for time in t:
    Q_omega_values_realpart = [np.real(Q_DR(time, momentum)) for momentum in k]
    Q_omega_values_imagpart = [np.imag(Q_DR(time, momentum)) for momentum in k]

    ax1.plot(k, Q_omega_values_realpart, label='geodesic length (time) = ''%.2f'''%time)
    ax2.plot(k, Q_omega_values_imagpart)

ax1.legend()
plt.show()