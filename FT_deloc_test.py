import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import quad, cumulative_trapezoid, simpson
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

a = 1
m = 1
phi_m = -10
t_flight1 = 10

#Apply measurment, delocalized state
E=1/(2*m*a**2)
#phase accumulated during first localized evolution
phi_1 = E*t_flight1/4
#ratio of sigma/sigma_d
ratio = 20
#width at end of first flight
sigma_squared = a**2*(1+(E*t_flight1)**2)
#half distance between slits, we set it to half the width of the initial packets, hence the distance is equal to the width of the packets
d=np.sqrt(sigma_squared)/2
#squared widht of slits in double slit squared position measurement
sigmad_squared = sigma_squared/ratio**2
#total phase
phi = phi_1 + phi_m

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

def IFT_evo(ftrans, k, x_values, t):

    #dispersion relation
    #omega = k**2/(2*m)
    #omega = np.sqrt(k**2 + m**2)

    ift = []
    for x in x_values:

        L_0 = 1
        gl = np.sqrt(np.abs(t**2-x**2))
        xi2 = gl**2/L_0**2
        f_xi = ((1+1/xi2)/(np.sqrt(xi2)*L_0))*((2-1/xi2)/(1+xi2))
        omega = 1j*f_xi*(1/(2*np.sqrt(1+1/xi2)))-(1/(2*m*np.sqrt(1+1/xi2)))*k**2

        integrand = ftrans*np.exp(1j*(k*x-omega*t))

        ift_x = complex_trapz(integrand)
        ift.append(ift_x)
    return ift

samples = 1000
#extent of integration and plotting, enlarge if ripple effects due to boundaries arise
extent = 30
x = np.linspace(-extent, extent, samples)
k = np.linspace(-extent, extent, samples)

psi_k = FT(x, k)
fig1, ax1 = plt.subplots()
ax1.plot(k, np.abs(psi_k))

fig2, ax2 = plt.subplots()

t_flight2 = 2
for t in np.linspace(0, t_flight2, 10):
    psi_xt = IFT_evo(psi_k, k, x, t)

    norm = np.sqrt(np.trapz(np.abs(psi_xt)**2))

    ax2.plot(x, np.abs(psi_xt)/norm, label='time={}'.format(t))

plt.legend()

time_elapsed = (time.perf_counter() - time_start)
print ("checkpoint %5.1f secs" % (time_elapsed))

plt.show()