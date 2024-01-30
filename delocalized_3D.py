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
#measurement induced phase
phi_m = -2

#values for position
x = np.linspace(-4, 4, 1001)

#time of flight before delocalization
t_flight1 = 10

#time of flight of delocalized state before measurement
t_flight2 = 1.5
time2 = np.linspace(0, t_flight2, 1001)

#Apply measurment, delocalized state
E=1/(2*m*a**2)
#phase accumulated during first localized evolution
phi_1 = E*t_flight1/4
print("The Wavepackets are moving towards eachother:" + str(phi_1+phi_m<0))
#ratio of sigma/sigma_d
ratio = 50
#width at end of first flight
sigma_squared = a**2*(1+(E*t_flight1)**2)
#half distance between slits, we set it to half the width of the initial packets, hence the distance is equal to the width of the packets
d=np.sqrt(sigma_squared)/2
print("Width of wavepacket before double slit: {}".format(2*d))
#squared widht of slits in double slit squared position measurement
sigmad_squared = sigma_squared/ratio**2
#total phase
phi = phi_1 + phi_m

Sigma = (sigma_squared*sigmad_squared)/(sigma_squared+sigmad_squared*(1-4*1j*phi))
D = d/sigmad_squared

#calculate normalization at every timestep, save in list
norm = []
for t in time2:
    def DS_squared_modulus(x, t=t):
        R = np.exp(-(1/4)*(x-D*Sigma)**2/(Sigma+(1j*t)/(2*m)))
        L = np.exp(-(1/4)*(x+D*Sigma)**2/(Sigma+(1j*t)/(2*m)))
        mod_squared = np.abs(R + L)**2
        return mod_squared

    norm_squared, res = integrate.quad(DS_squared_modulus, -np.inf, np.inf)
    norm.append(np.sqrt(norm_squared))
norm_t = np.array(norm)

def DS(x, t):
    R = np.exp(-(1/4)*(x-D*Sigma)**2/(Sigma+(1j*t)/(2*m)))
    L = np.exp(-(1/4)*(x+D*Sigma)**2/(Sigma+(1j*t)/(2*m)))
    return (R + L)

#create grid for plotting, calculate modulus and angle
X, T = np.meshgrid(x, time2)
mod = np.abs(DS(X, T))/norm_t[:,None]
phase = np.angle(DS(X, T))

# Creating figure
fig = plt.figure()
gs = mpl.gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[2, 1], wspace=0.5, hspace=0.5,height_ratios=[1])
ax = fig.add_subplot(gs[0], projection='3d')

#change scale of x axis for better displaying
scale_x = 2
scale_y = 1
scale_z = 1
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))

#plot modulus of wavefunction with phase as colour
norm_colour = mpl.colors.Normalize(vmin=phase.min(), vmax=phase.max())
DLS = ax.plot_surface(X, T, mod, rstride=2, cstride=2, facecolors=cm.hsv(norm_colour(phase)), edgecolor ='none')
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
ax2 = fig.add_subplot(gs[1])

ax2.plot(x, np.abs(DS(x, t_flight2))/norm_t[len(norm_t)-1])
ax2.set_title('Final timeslice')
ax2.set_xlabel('X')
ax2.set_ylabel('$|\psi|$')

time_elapsed = (time.perf_counter() - time_start)
print ("checkpoint %5.1f secs" % (time_elapsed))

plt.show()

#if __name__ == '__main__':
#    main()

