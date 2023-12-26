import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import quad, cumulative_trapezoid
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

def complex_trapezoidal(func):
    real_func = np.real(func)
    imag_func = np.imag(func)
    real_integral = cumulative_trapezoid(real_func)
    imag_integral = cumulative_trapezoid(imag_func)
    return real_integral[-1]-real_integral[0] + 1j*(imag_integral[-1]-imag_integral[0])

a = 1
m = 1

def FT(x, k_values):
    ft = []
    for k in k_values:
        def integrand(x):
            initial_state = np.exp(-(x**2/(4*a**2)))
            return initial_state*np.exp(-1j*k*x)

        ft_k = complex_quadrature(integrand, -np.inf, np.inf)
        ft.append(ft_k)
    return ft

def IFT_evo(ftrans, k, x_values, t_flight1):

    omega = k**2/(2*m)

    ift = []
    for x in x_values:
        integrand = ftrans*np.exp(1j*(k*x-omega*t_flight1))

        ift_x = complex_trapezoidal(integrand)
        ift.append(ift_x)
    return ift

samples = 1000
x = np.linspace(-10, 10, samples)
k = np.linspace(-10, 10, samples)

psi_k = FT(x, k)
plt.plot(k, np.abs(psi_k))
plt.show()


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



for t in range(11):
    psi_xt = IFT_evo(psi_k, k, x, t)
    plt.plot(x, np.abs(psi_xt)/600*(6/5), label='numerical', c='red')

    y = [ES(i, t) for i in x]
    plt.plot(x, np.abs(y), label='exact', c='black')

plt.legend()
plt.show()

time_elapsed = (time.perf_counter() - time_start)
print ("checkpoint %5.1f secs" % (time_elapsed))