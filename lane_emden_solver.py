# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 17:45:32 2025

@author: aleja
"""

# Code to solve the Lane-Emden equation through Runge-Kutta fourth order method

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Close plot
plt.close("all")

# Lane-Emden function
def lane_emden(xi, y, n):
    theta, phi = y
    first_der = phi
    second_der = -theta**(n) - (2/xi)*phi
    return ([first_der, second_der])

# Defining event where theta becomes negative (for r=R integration limit)
def theta_negative(xi, y, n):
    theta = y[0]
    return (theta)

theta_negative.direction = -1 # stop integrating for downwards direction (neg.)
theta_negative.terminal = True # Boolean True

# Limits of integration
xi0 = 0.0000001 # avoid zero
ximax = 50

# Boundary conditions
theta = 1
phi = 0

# Defining the possible n values
n = np.arange(0, 5.5, step = 0.5)


"Defining and calculating the values of the constants Rn, Mn, Dn, Bn"

def M_n(xi_1, der_xi_1):
    return (-(xi_1**2)*(der_xi_1))

def D_n(xi_1, der_xi_1):
    return (((-3/xi_1)*(der_xi_1))**(-1))

def B_n(xi_1, der_xi_1, n):
    parenthesis = (-(xi_1**2)*(der_xi_1))**(-2/3)
    return ((1/(n+1))*(parenthesis))

# Solving the Lane-Emden equation for different n values
for i in range(len(n)):
    solution = solve_ivp(lane_emden, [xi0, ximax], [theta, phi], 
                     t_eval=np.linspace(xi0, ximax, 100001), 
                     events=(theta_negative), args=(n[i],), atol=1e-10)
    plt.plot(solution.t, solution.y[0], label=r"$n$ = " + f"{n[i]}")
    print("n = " + f"{n[i]}\n" + "(xi_1, theta) = " + f"{solution.t[-1], solution.y[0][-1]}")
    print("(dtheta/dxi)xi_1 = " + f"{solution.y[1][-1]}")
    print("Values of the constants:\n" + f"R_{n[i]} = {solution.t[-1]}, \n" + f"M_{n[i]} = {M_n(solution.t[-1], solution.y[1][-1])}, \n" + f"D_{n[i]} = {D_n(solution.t[-1], solution.y[1][-1])}, \n" + f"B_{n[i]} = {B_n(solution.t[-1], solution.y[1][-1], n[i])}\n")
    print("")    
    
"Calculating the mass and radius of the non-relativistic white dwarf"

# Defining the solution for the White Dwarf
solution_wd = solve_ivp(lane_emden, [xi0, ximax], [theta, phi], 
                 t_eval=np.linspace(xi0, ximax, 100001), 
                 events=(theta_negative), args=(n[3],), atol=1e-10)

# Constants
me = 9.1093837015*10**(-31) # kg
mh = 1.660539*10**(-27) # kg
mu_e = 2 # electrons
h_bar = 1.0545718*10**(-34) # Js
K = ((h_bar**2)*(3*np.pi**2)**(2/3))/(5*me*(mh*mu_e)**(5/3)) # polytropic const.
G = 6.674*10**(-11) # Nm^2/kg^2

# Solar constants
solar_radius = 6.957*10**(8) # m
solar_mass = 1.98847*10**(30) # kg

# Central densities
rho_c1 = 5*10**8 # kg/m^3
rho_c2 = 10**9 # kg/m^3
rho_c3 = 5*10**9 # kg/m^3
rho_c = np.array([rho_c1, rho_c2, rho_c3])

# Defining the radius and the mass of the non-relativistic white dwarf

def radius(xi, rho_c, n=n[3]):
    alpha = np.sqrt(((n+1)*K)/(4*np.pi*G*rho_c**((n-1)/n)))
    return alpha*xi

def mass(rho_c, M_n, n=n[3]):
    alpha = np.sqrt(((n+1)*K)/(4*np.pi*G*rho_c**((n-1)/n)))
    return (4*np.pi*(alpha**3)*rho_c*M_n)

# In solar radius and masses
def radius_solar(radius):
    return radius/solar_radius

def mass_solar(mass):
    return mass/solar_mass

# Printing the values
print("Calculation of Radius and Masses for the different central densities:")
for i in range(len(rho_c)):
    print("rho_c = " + f"{rho_c[i]}")
    print("R_WD = " + f"{radius_solar(radius(solution_wd.t[-1], rho_c[i]))}" + " R_solar")
    print("M_WD = " + f"{mass_solar(mass(rho_c[i], M_n(solution_wd.t[-1], solution_wd.y[1][-1])))}" + " M_solar")
    print("")
    
"Parameters for plotting the physical properties of the white dwarf"

# Defining the mean density
def rho(rho_c, theta, n=n[3]):
    return (rho_c*theta**n)

def m(xi, der_xi, rho_c, n=n[3]):
    alpha = np.sqrt(((n+1)*K)/(4*np.pi*G*rho_c**((n-1)/n)))
    return (4*np.pi*(alpha**3)*rho_c*(-(xi**2)*(der_xi)))
    
# For first central density
log_density_1 = np.log10(rho(rho_c[0], solution_wd.y[0]))
mass_wd1 = mass_solar(m(solution_wd.t, solution_wd.y[1], rho_c[0]))
radius_wd1 = radius_solar(radius(solution_wd.t, rho_c[0]))

# For second central density
log_density_2 = np.log10(rho(rho_c[1], solution_wd.y[0]))
mass_wd2 = mass_solar(m(solution_wd.t, solution_wd.y[1], rho_c[1]))
radius_wd2 = radius_solar(radius(solution_wd.t, rho_c[1]))

# For third central density
log_density_3 = np.log10(rho(rho_c[2], solution_wd.y[0]))
mass_wd3 = mass_solar(m(solution_wd.t, solution_wd.y[1], rho_c[2]))
radius_wd3 = radius_solar(radius(solution_wd.t, rho_c[2]))

"Calculating the central density and radius for a M=0.5405 solar masses Sun as white dwarf"

def central_density(M, M_n, n):
    left = (4*np.pi*M_n)**2
    right = (((n+1)*K)/(4*np.pi*G))**3
    return (M**2)/(left*right)

def Radius(M, rho_c, M_n, R_n):
    lower = (4*np.pi*rho_c*M_n)
    return R_n*((M/lower)**(1/3))

# Values for the central densities and radius required
M = 0.5405*solar_mass # kg

central_density_value = central_density(M, M_n(solution_wd.t[-1], solution_wd.y[1][-1]), n=1.5)
radius_value = Radius(M, central_density_value, M_n(solution_wd.t[-1], solution_wd.y[1][-1]), solution_wd.t[-1])

print("")
print("These are the central densities and radius for the M=0.540 solar masses Sun: ")
print("Central Density value = " + f"{central_density_value}")
print("Radius value = " + f"{radius_value}")
print("")
print("")

# Values for the plotting of the White Dwarf Sun
log_density_sun_wd = np.log10(rho(central_density_value, solution_wd.y[0]))
mass_sun_wd = mass_solar(m(solution_wd.t, solution_wd.y[1], central_density_value))
radius_sun_wd = radius_solar(radius(solution_wd.t, central_density_value))

"Plotting snippet of code"

# Analytical solutions for n=0 and n=1
def analytical_n0_n1(xi):
    return ([1 - ((xi**2)/6), np.sin(xi)/xi])

# Defining number of points to plot
t_eval=np.linspace(xi0, ximax, 10001)

# Labeling and plotting axes
plt.figure(1)
#plt.plot(t_eval, analytical_n0_n1(t_eval)[0], color="black", linestyle="dashed", label="Analytical n = " + f"{n[0]}")
#plt.plot(t_eval, analytical_n0_n1(t_eval)[1], color="black", linestyle="dashed", label="Analytical n = " + f"{n[2]}")
plt.xlabel(r"$\xi$", fontsize=40)
plt.ylabel(r"$\theta$", fontsize=40)
plt.tick_params(direction="in", which="major", length=8, labelsize=36, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=36, top=True, right=True)
plt.xticks(np.arange(0, 11, step=1))
plt.yticks(np.arange(0.1, 1.0, step=0.1))
plt.ylim(0, 1.0)
plt.xlim(0.1, 11)
plt.legend(loc="best", prop={"size":26})
plt.tight_layout()
plt.show()

# Plotting log(p) vs r/R
plt.figure(2)
plt.plot(radius_wd1, log_density_1, color="mediumslateblue", label=r"$\rho_c = 5\times10^{8}\ \mathrm{[kg\,m^{-3}]}$")
plt.plot(radius_wd2, log_density_2, color="deepskyblue", label=r"$\rho_c = 1\times10^{9}\ \mathrm{[kg\,m^{-3}]}$")
plt.plot(radius_wd3, log_density_3, color="midnightblue", label=r"$\rho_c = 5\times10^{9}\ \mathrm{[kg\,m^{-3}]}$")
plt.xlabel(r"$r$/R$_{\odot}$", fontsize=40)
plt.ylabel(r"$\log(\rho) \,$ [kg$\,$m$^{-3}$]", fontsize=40)
plt.tick_params(direction="in", which="major", length=8, labelsize=36, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=36, top=True, right=True)
plt.yticks(np.arange(5, 10, step=1))
plt.ylim(4, 10)
plt.xlim(0.0015)
plt.legend(loc="best", prop={"size":30})
plt.tight_layout()
plt.show()

# Plotting m/M vs r/R
plt.figure(3)
plt.plot(radius_wd1, mass_wd1, color="goldenrod", label=r"$\rho_c = 5\times10^{8}\ \mathrm{[kg\,m^{-3}]}$")
plt.plot(radius_wd2, mass_wd2, color="red", label=r"$\rho_c = 1\times10^{9}\ \mathrm{[kg\,m^{-3}]}$")
plt.plot(radius_wd3, mass_wd3, color="darkred", label=r"$\rho_c = 5\times10^{9}\ \mathrm{[kg\,m^{-3}]}$")
plt.xlabel(r"$r$/R$_{\odot}$", fontsize=40)
plt.ylabel(r"$m$/M$_{\odot}$", fontsize=40)
plt.tick_params(direction="in", which="major", length=8, labelsize=36, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=36, top=True, right=True)
plt.yticks(np.arange(0.1, 1.1, step=0.1))
plt.ylim(0, 1.1)
plt.xlim(0.001, 0.019)
plt.legend(loc="best", prop={"size":30})
plt.tight_layout()
plt.show()

# Plotting the 0.540 solar masses Sun

# Plotting log(p) vs r/R
plt.figure(4)
plt.plot(radius_sun_wd, log_density_sun_wd, color="slategrey")
plt.xlabel(r"$r$/R$_{\odot}$", fontsize=40)
plt.ylabel(r"$\log(\rho) \,$ [kg$\,$m$^{-3}$]", fontsize=40)
plt.tick_params(direction="in", which="major", length=8, labelsize=36, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=36, top=True, right=True)
plt.yticks(np.arange(4, 10, step=1))
plt.ylim(3, 10)
plt.xlim(0.0015)
plt.legend(loc="best", prop={"size":30})
plt.tight_layout()
plt.show()

# Plotting m/M vs r/R
plt.figure(5)
plt.plot(radius_sun_wd, mass_sun_wd, color="slategrey")
plt.xlabel(r"$r$/R$_{\odot}$", fontsize=40)
plt.ylabel(r"$m$/M$_{\odot}$", fontsize=40)
plt.tick_params(direction="in", which="major", length=8, labelsize=36, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=36, top=True, right=True)
plt.yticks(np.arange(0.1, 1.0, step=0.1))
plt.ylim(0, 1)
plt.xlim(0.001, 0.015)
plt.legend(loc="best", prop={"size":30})
plt.tight_layout()
 
