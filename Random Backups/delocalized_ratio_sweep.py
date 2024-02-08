import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib as mpl
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy.integrate as integrate
import time

time_start = time.perf_counter()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

v = [6**2, 10**2, 20**2, 50**2]
print(v)
for ratio in v: 
    #mass of the object
    m=1
    #size of the ground state of an harmonic oscillator
    a=1
    #measurement induced phase
    phi_m = -2
    #values for position
    x = np.linspace(-10, 10, 1001)
    #time of flight before delocalization
    t_flight1 = 10
    #time of flight of delocalized state before measurement
    t_flight2 = 1
    #Energy of HO GS
    E=1/(2*m*a**2)
    #phase accumulated during first localized evolution
    phi_1 = E*t_flight1/4
    print("The Wavepackets are moving towards eachother:" + str(phi_1+phi_m<0))
    #width at end of first flight
    sigma_squared = a**2*(1+(E*t_flight1)**2)
    print(sigma_squared)
    #half distance between slits, we set it to half the width of the initial packets, hence the distance is equal to the width of the packets
    d=np.sqrt(sigma_squared)/2
    #squared widht of slits in double slit squared position measurement
    sigmad_squared = sigma_squared/ratio
    #total phase
    phi = phi_1 + phi_m

    Sigma = (sigma_squared*sigmad_squared)/(sigma_squared+sigmad_squared*(1-4*1j*phi))
    D = d/sigmad_squared

    #calculate normalization
    def DS_squared_modulus(x):
        R = np.exp(-(1/4)*(x-D*Sigma)**2/(Sigma+(1j*t_flight2)/(2*m)))
        L = np.exp(-(1/4)*(x+D*Sigma)**2/(Sigma+(1j*t_flight2)/(2*m)))
        mod_squared = np.abs(R + L)**2
        return mod_squared

    norm_squared, res = integrate.quad(DS_squared_modulus, -np.inf, np.inf)
    norm = np.sqrt(norm_squared)

    def DS(x):
        R = np.exp(-(1/4)*(x-D*Sigma)**2/(Sigma+(1j*t_flight2)/(2*m)))
        L = np.exp(-(1/4)*(x+D*Sigma)**2/(Sigma+(1j*t_flight2)/(2*m)))
        return (R + L)

    #create grid for plotting, calculate modulus and angle
    mod = np.abs(DS(x))/norm
    phase = np.angle(DS(x))

    #plotting
    ax.plot(x, np.abs(DS(x))/norm, label='$\sigma/\sigma_d=$'+str(np.sqrt(ratio)))
    ax.set_title('Final timeslice')
    ax.set_xlabel('X')
    ax.set_ylabel('$|\psi|$')

time_elapsed = (time.perf_counter() - time_start)
print ("checkpoint %5.1f secs" % (time_elapsed))

ax.legend()
plt.show()

#if __name__ == '__main__':
#    main()

