import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import time

time_start = time.perf_counter()

def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j*imag_integral[0]

def complex_trapz(func):
    real_func = np.real(func)
    imag_func = np.imag(func)
    real_integral = np.trapz(real_func)
    imag_integral = np.trapz(imag_func)
    return real_integral + 1j*imag_integral

#Type of dispersion relation tu use for time evolution
DR = 'Q'

#Width of initial state gaussian
a = 1
#mass
m = 1
#measurement induced phase, loosely corresponds to the speed of ricombination IMPORTANT!
phi_m = -2
#time of flight before delocalization
t_flight1 = 10

#Energy of initial state
E=1/(2*m*a**2)
#phase accumulated during first localized evolution
phi_1 = E*t_flight1/4
#ratio of sigma/sigma_d
ratio = 50
#width at end of first flight
sigma_squared = a**2*(1+(E*t_flight1)**2)
#half distance between slits, we set it to half the width of the initial packets, hence the distance is equal to the width of the packets
d=np.sqrt(sigma_squared)/2
print('slit distance =' + str(2*d))
#squared widht of slits in double slit squared position measurement
sigmad_squared = sigma_squared/ratio**2
print('slit width =' + str(np.sqrt(sigmad_squared)))
#total phase, must be negative for convergence of wavepackets
phi = phi_1 + phi_m

#Defines the delocalized state in position space after measurment 
#of squared position and computes the spatial fourier trasform
def FT(x, k_values):
    ft = []
    for k in k_values:
        def integrand(x):
            a_coeff = (sigma_squared+sigmad_squared*(1-4j*phi))/(4*sigma_squared*sigmad_squared)
            b_coeff = -d/(2*sigmad_squared)
            c_coeff = d**2/(4*sigmad_squared)
            L = np.exp(-a_coeff*x**2-b_coeff*x-c_coeff)
            R = np.exp(-a_coeff*x**2+b_coeff*x-c_coeff)
            initial_state = L + R
            return initial_state*np.exp(-1j*k*x)

        ft_k = complex_quadrature(integrand, -np.inf, np.inf)
        ft.append(ft_k)
    return ft

#Computes the inverse fourier transform after evolving each plane wave component with the correct time harmonic function
def IFT_evo(ftrans, k, x_values, t):

    #dispersion relation
    if DR == 'SCH':
        omega = k**2/(2*m)
    
    elif DR == 'KG':
        omega = np.sqrt(k**2 + m**2)

    ift = []
    for x in x_values:

        integrand = ftrans*np.exp(1j*(k*x-omega*t))

        ift_x = complex_trapz(integrand)
        ift.append(ift_x)
    return ift

#number of points for integration and plotting
samples = 10000
#extent of integration and plotting, enlarge if ripple effects due to boundaries arise
k_extent = 100
x_extent = 100

x = np.linspace(-x_extent, x_extent, samples)
k = np.linspace(-k_extent, k_extent, samples)

#Compute FT
psi_k = FT(x, k)

#Define plot, plot FT
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
ax1.plot(k, np.abs(psi_k))
ax1.set_title('Momentum space')
ax1.set_xlabel('k')
ax1.set_ylabel('$\psi (k)$')

#Time of flight after delocalization
t_flight2 = 3
#Compute the wavefunction evolved for different times
eps = 0.001
for t in np.linspace(0+eps, t_flight2, 5):
    psi_xt = IFT_evo(psi_k, k, x, t)

    #Compute norm of wavefunction to be used for normalization below
    norm = np.sqrt(np.trapz(np.abs(psi_xt)**2))

    print(norm)

    ax2.plot(x, np.abs(psi_xt)/norm, label='time=%.2f'%t)

ax2.set_title('Position space')
ax2.set_xlabel('x')
ax2.set_ylabel('$\psi (x, t)$')
ax2.legend()
ax2.set_xlim([-4, 4])

time_elapsed = (time.perf_counter() - time_start)
print ("checkpoint %5.1f secs" % (time_elapsed))

plt.show()