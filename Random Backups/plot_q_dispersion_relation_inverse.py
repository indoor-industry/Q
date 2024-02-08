import numpy as np
import matplotlib.pyplot as plt

m = 5
omega = np.linspace(0, 10, 10001)

fig, (ax1, ax2) = plt.subplots(1, 2)

#KG dispersion relation
def KG_DR(omega):
    k = np.emath.sqrt(omega**2-m**2)
    return k

KG_values = [KG_DR(energy) for energy in omega]

ax1.plot(omega, np.real(KG_values), label='KG real', color='red')
ax1.plot(omega, np.imag(KG_values), label='KG imag', color='black')
ax1.plot(omega, omega, linestyle='dashed', label='m=0')

#q-metric dispersion relation
def Q_DR(t, omega):
    dim = 4
    L_0 = 1
    gl = t
    xi = (L_0/gl)**2
    T_squared = 1 + xi
    g = -dim*T_squared**(-1)*(L_0**2/gl**3)
    k = T_squared*np.emath.sqrt(omega**2+1j*g*omega-T_squared**(-1)*m**2)
    return k

time = 20
Q_values = [Q_DR(time, energy) for energy in omega]
ax2.plot(omega, np.real(Q_values), color='red')
ax2.plot(omega, np.imag(Q_values), color='black')
ax2.plot(omega, omega, linestyle='dashed')

ax2.set_title('q-metric $k(\omega)$')
ax2.set_ylabel('k')
ax2.set_xlabel('$\omega$')

ax1.set_title('KG $k(\omega)$')
ax1.set_ylabel('k')
ax1.set_xlabel('$\omega$')
ax1.legend()
plt.show()