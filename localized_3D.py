import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib as mpl
import time

time_start = time.perf_counter()

#mass of the object
m=10
#size of the ground state of an harmonic oscillator
a=1

#values for position
x = np.linspace(-10, 10, 1001)

#ground state
def GS(x, a):
    return (1/(2*np.pi*a**2))**(1/4)*np.exp(-(x**2/(4*a**2)))

#evolved state
def ES(x, t):
    #GS energy
    E=1/(2*m*a**2)
    #time dependent width
    sigma_squared = a**2*(1+(E*t)**2)
    #common phase factor
    b = (1-1j*E*t)

    normalization_prefactor = (2*np.pi*sigma_squared)**(-1/4)
    main_exp = np.exp(-(x**2/(4*sigma_squared))*b)
    return normalization_prefactor*main_exp

#time of flight before delocalization
t_flight1 = 50
time1 = np.linspace(0, t_flight1, 101)

print("Width of wavepacket before double slit: {}".format(np.sqrt(a**2*(1+((1/(2*m*a**2))*t_flight1)**2))))

X, T = np.meshgrid(x, time1)
mod = np.abs(ES(X, T))
phase = np.angle(ES(X, T))

# Creating figure
fig = plt.figure()
gs = mpl.gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[2, 1], wspace=0.5, hspace=0.5,height_ratios=[1])
ax = fig.add_subplot(gs[0], projection='3d')

scale_x = 2
scale_y = 1
scale_z = 1
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))

norm_colour = mpl.colors.Normalize(vmin=phase.min(), vmax=phase.max())
LS = ax.plot_surface(X, T, mod, rstride=2, cstride=2, facecolors=cm.hsv(norm_colour(phase)), edgecolor ='none')
colbar = cm.ScalarMappable(cmap=plt.cm.hsv, norm=norm_colour)
cbar = fig.colorbar(colbar, ax = ax, orientation='horizontal', fraction=0.05)
cbar.set_ticks([-np.pi, -np.pi/2 , 0, np.pi/2, np.pi])
cbar.set_ticklabels(['$-\pi$', '$-\pi/2$', 0, '$\pi/2$', '$\pi$'])
cbar.set_label('Phase')
ax.set_title('Localized state')
ax.set_xlabel('X')
ax.set_ylabel('T')
ax.set_zlabel('$|\psi|$')

#fig2 = plt.figure()
ax2 = fig.add_subplot(gs[1])

ax2.plot(x, np.abs(ES(x, t_flight1)))
ax2.set_title('Final timeslice')
ax2.set_xlabel('X')
ax2.set_ylabel('$|\psi|$')

time_elapsed = (time.perf_counter() - time_start)
print ("checkpoint %5.1f secs" % (time_elapsed))

plt.show()