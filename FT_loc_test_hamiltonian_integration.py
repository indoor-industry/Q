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
m = 100

def FT(x, k_values):
    ft = []
    for k in k_values:
        def integrand(x):
            initial_state = np.exp(-(x**2/(4*a**2)))
            return initial_state*np.exp(1j*k*x)

        ft_k = complex_quadrature(integrand, -np.inf, np.inf)
        ft.append(ft_k)
    return ft

def IFT_evo(ftrans, k, x_values, t_flight1):

    #q-metric dispersion relation
    dim = 4
    L_0 = 1
    omega_t_int = []
    for momentum in k:
        def omega_t_integrand(gl):
            xi = (L_0/gl)**2
            T_squared = 1 + xi
            g = (((dim-1)/gl)*(1-T_squared**(-2))-dim*T_squared**(-1)*(L_0**2/gl**3))
            omega_t_integrand = 0.5*(1j*g - np.sqrt(-g**2 + 4*T_squared**(-1)*(T_squared**(-1)*momentum**2 + m**2)))
            return omega_t_integrand
        omega_t_integrated = complex_quadrature(omega_t_integrand, 0.1, t)
        omega_t_int.append(omega_t_integrated)

    ift = []
    for x in x_values:

        integrand = ftrans*np.exp(-1j*(k*x-omega_t_int))

        ift_x = complex_trapz(integrand)
        ift.append(ift_x)
    return ift

samples = 10000
extent = 10
x = np.linspace(-extent, extent, samples)
k = np.linspace(-extent, extent, samples)

psi_k = FT(x, k)
plt.plot(k, np.abs(psi_k))
plt.show()

t_flight = 200
for t in np.linspace(0.1, t_flight, 5):
    psi_xt = IFT_evo(psi_k, k, x, t)

    norm = np.sqrt(np.trapz(np.abs(psi_xt)**2))

    print(norm)

    plt.plot(x, np.abs(psi_xt)/norm, label='time={}'.format(t))

plt.legend()
plt.show()

time_elapsed = (time.perf_counter() - time_start)
print ("checkpoint %5.1f secs" % (time_elapsed))