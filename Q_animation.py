import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import time
from celluloid import Camera

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

def integrand(x):
    evolved_ground_state = (1/(2*np.pi*sigma_squared)**(-1/4))*np.exp(-(x**2/sigma_squared)*(0.25-1j*phi_1))
    measurement_induced_phase = np.exp(1j*phi_m*(x**2/sigma_squared))
    L = measurement_induced_phase*np.exp(-(x+d)**2/(4*sigmad_squared))#*evolved_ground_state
    R = measurement_induced_phase*np.exp(-(x-d)**2/(4*sigmad_squared))#*evolved_ground_state
    initial_state = L + R
    return initial_state

#Defines the delocalized state in position space after measurment 
#of squared position and computes the spatial fourier trasform
def FT(x, k_values):
    ft = []
    for k in k_values:
        ft_k = complex_trapz(integrand(x)*np.exp(-1j*k*x))#, -np.inf, np.inf)
        ft.append(ft_k)
    return ft

#Computes the inverse fourier transform after evolving each plane wave component with the correct time harmonic function
def IFT_evo(ftrans, k, x_values, t):

    #q-metric dispersion relation
    dim = 4
    L_0 = 1
    omega_t_int = []
    for momentum in k:
        def omega_t_integrand(gl):
            xi = (L_0/gl)**2
            T_squared = 1 + xi
            T_sq_inv = 1/T_squared
            g = -dim*T_sq_inv*(L_0**2/gl**3)
            #ADDED A MINUS IT DOESN'T WORK OTHERWISE :'(
            omega_t_integrand = -(T_sq_inv*momentum**2 + (1-T_squared)*m**2 - 1j*m*T_squared*g)/(T_squared*(g - 2*1j*m))
            #omega_t_integrand = 1j*(momentum**2/(2*m))
            return omega_t_integrand
        omega_t_integrated = complex_quadrature(omega_t_integrand, 0.01, t)
        omega_t_int.append(omega_t_integrated)

    ift = []
    for x in x_values:

        integrand = ftrans*np.exp(1j*k*x-omega_t_int)

        ift_x = complex_trapz(integrand)
        ift.append(ift_x)
    return ift

#number of points for integration and plotting
samples = 1000
#extent of integration and plotting, enlarge if ripple effects due to boundaries arise
k_extent = 30
x_extent = 5

x = np.linspace(-x_extent, x_extent, samples)
k = np.linspace(-k_extent, k_extent, samples)

#initial state
IS = [integrand(pos) for pos in x]

#Compute FT
psi_k = FT(x, k)

#Define plot
fig2, ax2 = plt.subplots()
camera = Camera(fig2)

#Time of flight after delocalization
t_flight2 = 5
#Compute the wavefunction evolved for different times
eps = 0.001
for t in np.linspace(0+eps, t_flight2, 200):
    psi_xt = IFT_evo(psi_k, k, x, t)

    #Compute norm of wavefunction to be used for normalization below
    norm = np.sqrt(np.trapz(np.abs(psi_xt)**2))

    print(norm)

    plot = ax2.plot(x, np.abs(psi_xt)/norm, color = 'black')
    ax2.legend(plot, [f'time = {t:.2f}'])
    ax2.set_title('Position space')
    ax2.set_xlabel('x')
    ax2.set_ylabel('$\psi (x, t)$')
    ax2.set_xlim([-2*d, 2*d])
    camera.snap()

animation = camera.animate(interval = 2)
animation.save('plots/Q_evo.gif')

time_elapsed = (time.perf_counter() - time_start)
print ("checkpoint %5.1f secs" % (time_elapsed))