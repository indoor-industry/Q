import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.axes3d import Axes3D

#mass of the object
m=1
#size of the ground state of an harmonic oscillator
a=1
#energy of ground state hbar=1
E=1/(2*m*a**2)

#values for position
x = np.linspace(-10, 10, 100)

#ground state
def GS(x, a):
    return (1/(2*np.pi*a**2))**(1/4)*np.exp(-(x**2/(4*a**2)))

#evolved state
def ES(x, t):
    sigma_squared = a**2*(1+(E*t)**2)
    return (2*np.pi*sigma_squared)**(-1/4)*np.exp(-((x**2*(1-1j*E*t))/(4*sigma_squared)))

#time of flight before delocalization
t_flight1 = 10
time1 = np.linspace(0, t_flight1, 50)

#Apply measurment, delocalized state, not normalized
sigma_squared = a**2*(1+(E*t_flight1)**2)
sigmad_squared = sigma_squared/10
phi = -2 
d=10
def DS(x, t):
    D = sigma_squared+sigmad_squared-1j*sigmad_squared*(E*t_flight1+phi)
    L = np.exp(-(((d*sigma_squared)/(2*D)-x)**2)/(4*((sigma_squared*sigmad_squared)/D+1j*E*a**2*t)))
    R = np.exp(-(((-d*sigma_squared)/(2*D)-x)**2)/(4*((sigma_squared*sigmad_squared)/D+1j*E*a**2*t)))
    return R + L

X, T = np.meshgrid(x, time1)
wf = np.abs(ES(X, T))
phase = np.imag(ES(X, T))

# Creating figure
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

scale_x = 2
scale_y = 1
scale_z = 1
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))

DLS = ax.plot_surface(X, T, wf, cmap='seismic', facecolors=cm.seismic(phase) , edgecolor ='none')
fig.colorbar(DLS, ax = ax, shrink = 0.5, aspect = 5)
ax.set_title('Delocalized state')
ax.set_xlabel('X')
ax.set_ylabel('T')
ax.set_zlabel('$|\psi|$')

fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)

ax2.plot(x, np.abs(ES(x, t_flight1)))
ax2.set_title('Final timeslice')
ax2.set_xlabel('X')
ax2.set_ylabel('$|\psi|$')

plt.show()