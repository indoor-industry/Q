import numpy as np
import matplotlib.pyplot as plt

m = 1

eps=0.001
t_few = np.linspace(eps, 2, 5)
t_many = np.linspace(0.15, 3, 10000)
k = np.linspace(-100, 100, 10001)

fig, (ax1, ax2) = plt.subplots(1, 2)

#KG dispersion relation
def KG_DR(k):
    omega = np.sqrt(k**2+m**2)
    return omega

#SCHR dispersion relation
def SCHR_DR(k):
    omega = k**2/(2*m)
    return omega

KG_omega_values = [np.abs(KG_DR(momentum)) for momentum in k]
SCHR_DR_omega_values = [np.abs(SCHR_DR(momentum)) for momentum in k]

ax1.plot(k, KG_omega_values, label='KG', color='black')

#q-metric dispersion relation
def Q_DR(t, k):
    dim = 4
    L_0 = 1
    gl = t
    xi = (L_0/gl)**2
    T_squared = 1 + xi
    g = -dim*T_squared**(-1)*(L_0**2/gl**3)
    omega = 0.5*(-1j*g + np.emath.sqrt(-g**2 + 4*T_squared**(-1)*(T_squared**(-1)*k**2 + m**2)))
    return omega

def Q_DR_lowk(t, k):
    dim = 4
    L_0 = 1
    gl = t
    xi = (L_0/gl)**2
    T_squared = 1 + xi
    g = -dim*T_squared**(-1)*(L_0**2/gl**3)
    squareroot = np.emath.sqrt(-g**2 + 4*T_squared**(-1)*m**2)
    omega = 0.5*(-1j*g + squareroot)+(T_squared**(-2)*k**2)/squareroot
    return omega

for time in t_few:
    Q_omega_values_real = [np.real(Q_DR(time, momentum)) for momentum in k]
    ax1.plot(k, Q_omega_values_real, label='GL (time) = ''%.2f'''%time)

Q_omega_values_imagpart = [np.imag(Q_DR(time, 10)) for time in t_many]
ax2.plot(t_many, Q_omega_values_imagpart)
ax2.set_title('Imaginary part of $\omega$')
ax2.set_xlabel('$\sigma$')
ax2.set_ylabel('Im($\omega$)')

ax1.set_title('Dispersion relation vs $\sigma$')
ax1.set_xlabel('k')
ax1.set_ylabel('Re($\omega$)')
ax1.legend()
plt.show()