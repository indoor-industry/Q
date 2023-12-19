import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib as mpl
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy.integrate as integrate
import time

time_start = time.perf_counter()

#mass of the object
m=1
#size of the ground state of an harmonic oscillator
a=1
#energy of ground state hbar=1
E=1/(2*m*a**2)

#values for position
x = np.linspace(-100, 100, 1001)

#ground state of harmonic oscillator
def GS(x, a):
    return (1/(2*np.pi*a**2))**(1/4)*np.exp(-(x**2/(4*a**2)))

#evolving state of harmonic oscilator
def ES(x, t):
    sigma_squared = a**2*(1+(E*t)**2)
    return (2*np.pi*sigma_squared)**(-1/4)*np.exp(-((x**2*(1-1j*E*t))/(4*sigma_squared)))

#time of flight before delocalization
t_flight1 = 4
time1 = np.linspace(0, t_flight1, 50)

#time of flight of delocalized state before measurement
t_flight2 = 50
time2 = np.linspace(0, t_flight2, 1001)

#Apply measurment, delocalized state
sigma_squared = a**2*(1+(E*t_flight1)**2)
sigmad_squared = sigma_squared/10
phi = -2
d=10
#calculate normalization at every timestep, save in list
norm = []
for t in time2:
    def DS_squared_modulus(x, t=t):
        D = sigma_squared+sigmad_squared-1j*sigmad_squared*(E*t_flight1+phi)
        L = np.exp(-(((d*sigma_squared)/(2*D)-x)**2)/(4*((sigma_squared*sigmad_squared)/D+1j*E*a**2*t)))
        R = np.exp(-(((-d*sigma_squared)/(2*D)-x)**2)/(4*((sigma_squared*sigmad_squared)/D+1j*E*a**2*t)))
        mod_squared = np.abs(R + L)**2
        return mod_squared

    norm_squared, res = integrate.quad(DS_squared_modulus, -np.inf, np.inf)
    norm.append(np.sqrt(norm_squared))
norm_t = np.array(norm)

def DS(x, t):
    D = sigma_squared+sigmad_squared-1j*sigmad_squared*(E*t_flight1+phi)
    L = np.exp(-(((d*sigma_squared)/(2*D)-x)**2)/(4*((sigma_squared*sigmad_squared)/D+1j*E*a**2*t)))
    R = np.exp(-(((-d*sigma_squared)/(2*D)-x)**2)/(4*((sigma_squared*sigmad_squared)/D+1j*E*a**2*t)))
    return (R + L)

#create grid for plotting, calculate modulus and angle
X, T = np.meshgrid(x, time2)
mod = np.abs(DS(X, T))/norm_t[:,None]
phase = np.angle(DS(X, T))

# Creating figure
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

#change scale of x axis for better displaying
scale_x = 2
scale_y = 1
scale_z = 1
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))

#plot modulus of wavefunction with phase as colour
norm_colour = mpl.colors.Normalize(vmin=phase.min(), vmax=phase.max())
DLS = ax.plot_surface(X, T, mod, rstride=5, cstride=5, facecolors=cm.hsv(norm_colour(phase)), edgecolor ='none')
colbar = cm.ScalarMappable(cmap=plt.cm.hsv, norm=norm_colour)
cbar = fig.colorbar(colbar, ax = ax, orientation='horizontal', fraction=0.05)
cbar.set_ticks([-np.pi, -np.pi/2 , 0, np.pi/2, np.pi])
cbar.set_ticklabels(['$-\pi$', '$-\pi/2$', 0, '$\pi/2$', '$\pi$'])
cbar.set_label('Phase')
ax.set_title('Delocalized state')
ax.set_xlabel('X')
ax.set_ylabel('T')
ax.set_zlabel('$|\psi|$')

#plot final wavefunction shape at measurment (t=t_flight2)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)

ax2.plot(x, np.abs(DS(x, t_flight2))/norm_t[len(norm_t)-1])
ax2.set_title('Final timeslice')
ax2.set_xlabel('X')
ax2.set_ylabel('$|\psi|$')

time_elapsed = (time.perf_counter() - time_start)
print ("checkpoint %5.1f secs" % (time_elapsed))

plt.show()

#if __name__ == '__main__':
#    main()

