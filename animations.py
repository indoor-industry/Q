import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

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

ground_state = [GS(i, a) for i in x]

#plot ground state
#fig1, axs1 = plt.subplots(3)
#fig1.suptitle('Ground state')
#realGS = np.real(ground_state)
#imagGS = np.imag(ground_state)
#modGS = (np.abs(ground_state))**2
#axs1[0].plot(x, realGS, color='black')
#axs1[1].plot(x, imagGS, color='black')
#axs1[2].plot(x, modGS, color='black')
#plt.show()

#evolved state
def ES(x, a, t):
    sigma_squared = a**2*(1+(E*t)**2)
    return (2*np.pi*sigma_squared)**(-1/4)*np.exp(-((x**2*(1-1j*E*t))/(4*sigma_squared)))

#time of flight before delocalization
t_flight1 = 4
time1 = np.linspace(0, t_flight1, 50)

fig2, axs2 = plt.subplots(3)
fig2.suptitle('Evolving ground state')
camera = Camera(fig2)
for t in time1:
    #values of evolved state
    y = [ES(i, a, t) for i in x]
    #real, imaginary and absolute value parts
    realES = np.real(y)
    imagES = np.imag(y)
    modES = np.abs(y)
    #plot
    axs2[0].set_title('real part')
    axs2[0].plot(x, realES, color='black')
    axs2[1].set_title('imaginary part')
    axs2[1].plot(x, imagES, color='black')
    axs2[2].set_title('magnitude')
    axs2[2].plot(x, modES, color='black')
    camera.snap()
animation = camera.animate()
animation.save('localized_evolution.gif', writer = 'imagemagick')

#Apply measurment, delocalized state, not normalized
sigma_squared = a**2*(1+(E*t_flight1)**2)
sigmad_squared = sigma_squared/10
phi = -2 
d=10
def DS(x, a, t):
    D = sigma_squared+sigmad_squared-1j*sigmad_squared*(E*t_flight1+phi)
    L = np.exp(-(((d*sigma_squared)/(2*D)-x)**2)/(4*((sigma_squared*sigmad_squared)/D+1j*E*a**2*t)))
    R = np.exp(-(((-d*sigma_squared)/(2*D)-x)**2)/(4*((sigma_squared*sigmad_squared)/D+1j*E*a**2*t)))
    return R + L

#time of flight before delocalization
t_flight2 = 10
time2 = np.linspace(0, t_flight2, 50)

fig3, axs3 = plt.subplots(3)
fig3.suptitle('Evolving delocalized state')
camera2 = Camera(fig3)
for t in time2:
    #values of evolved state
    y = [DS(i, a, t) for i in x]
    #real, imaginary and absolute value parts
    realDS = np.real(y)
    imagDS = np.imag(y)
    modDS = np.abs(y)
    #plot
    axs3[0].set_title('real part')
    axs3[0].plot(x, realDS, color='black')
    axs3[1].set_title('imaginary part')
    axs3[1].plot(x, imagDS, color='black')
    axs3[2].set_title('magnitude')
    axs3[2].plot(x, modDS, color='black')
    camera2.snap()
animation2 = camera2.animate()
animation2.save('delocalized_evolution.gif', writer = 'imagemagick')
