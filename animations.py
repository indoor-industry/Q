import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from scipy import integrate
import time

time_start = time.perf_counter()

#mass of the object
m=1
#size of the ground state of an harmonic oscillator
a=1
#energy of ground state hbar=1
E=1/(2*m*a**2)

#values for position
x = np.linspace(-10, 10, 1000)

#ground state
def GS(x):
    return (1/(2*np.pi*a**2))**(1/4)*np.exp(-(x**2/(4*a**2)))

ground_state = [GS(i) for i in x]

#evolved state
def ES(x, t):
    sigma_squared = a**2*(1+(E*t)**2)
    return (2*np.pi*sigma_squared)**(-1/4)*np.exp(-((x**2*(1-1j*E*t))/(4*sigma_squared)))

#time of flight before delocalization
t_flight1 = 10
time1 = np.linspace(0, t_flight1, 200)

fig2, axs2 = plt.subplots()
fig2.suptitle('Evolving ground state')
camera = Camera(fig2)
for t in time1:
    #values of evolved state
    y = [ES(i, t) for i in x]
    #real, imaginary and absolute value parts
    modES = np.abs(y)
    #plot
    axs2.set_title('Localized')
    axs2.plot(x, modES, color='black')
    camera.snap()
animation = camera.animate()
animation.save('plots/localized_evolution.gif', writer = 'imagemagick')

#Apply measurment, delocalized state, not normalized
sigma_squared = a**2*(1+(E*t_flight1)**2)
ratio = 50
sigmad_squared = sigma_squared/ratio**2
phi = -2
d = np.sqrt(sigma_squared)/2
def DS(x, t):
    D = sigma_squared+sigmad_squared-1j*sigmad_squared*(E*t_flight1+phi)
    L = np.exp(-(((d*sigma_squared)/(2*D)-x)**2)/(4*((sigma_squared*sigmad_squared)/D+1j*E*a**2*t)))
    R = np.exp(-(((-d*sigma_squared)/(2*D)-x)**2)/(4*((sigma_squared*sigmad_squared)/D+1j*E*a**2*t)))
    return R + L

#time of flight before delocalization
t_flight2 = 2
time2 = np.linspace(0, t_flight2, 200)

fig3, axs3 = plt.subplots()
fig3.suptitle('Evolving delocalized state')
camera2 = Camera(fig3)
for t in time2:    
    #numerical normalization for each timeslice
    def DS_t_mod_squared(x):
        return np.abs(DS(x, t=t))**2

    normalization_squared, res = integrate.quad(DS_t_mod_squared, -np.inf, np.inf)
    normalization = np.sqrt(normalization_squared)

    #values of evolved state
    y = [DS(i, t) for i in x]
    #real, imaginary and absolute value parts
    modDS = np.abs(y)/normalization
    #plot
    axs3.set_title('magnitude')
    axs3.plot(x, modDS, color='black')
    camera2.snap()
animation2 = camera2.animate()
animation2.save('plots/delocalized_evolution.gif', writer = 'imagemagick')

time_elapsed = (time.perf_counter() - time_start)
print ("checkpoint %5.1f secs" % (time_elapsed))