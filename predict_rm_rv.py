#!/usr/bin/env python
# coding: utf-8

# In[137]:


# import packages
import numpy as np
import math as math
import pandas as pd
from astropy import units as u
import astropy.constants as const

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# get_ipython().run_line_magic('matplotlib', 'inline')

def get_a(mstar, m_p, P):
    """
    Calculate semi-major axis, a, using Kepler's Third law
    
    Inputs:
        mstar, stellar mass
        m_p, planet mass, 
        P, planet period
    Returns:
        a, semi-major axis (AU)
    
    """
    # period from kepler's third law
    # P = np.sqrt(((4*np.pi**2) / (const.G * mstar)) * a**3 )  
    # solve for a:
    
    a = (((const.G * (mstar + m_p) * P**2) / (4*np.pi**2))**(1/3)).to(u.AU)
    return a

def get_incl(b, a, rstar, e, omega):
    """
    For a given impact parameter, b, get the inclination.
    
    Inputs: 
      b, float, impact parameter (< 1, planet transits)
      a, Quantity [AU], semi-major axis
      rstar, Quantity [u.R_sun], planet radius
      e, float, (0 - circular)
      omega, float, (pi/2 - observer at periastron)
    
    Returns:
      incl, float, angle of orbital plane wrt reference plane
    
    """
    #b = ((a * math.cos(incl) / rstar)*((1 - e**2)/(1 + e * 
    #    math.sin(omega)))).decompose() # assume planet is passing right 
    #    through the middle of the disk of the star
    incl = math.acos((b / a) * rstar * (1 + e * math.sin(omega)) / (1 - e**2))
    return incl


def kepler_solver(n, P, nP, theta0, r0, ):
    """
    Solve for r, theta at range of times, t
    
    Inputs:
      n,      int, number of time steps
      P,      Quantity, period (days)
      nP,     number of periods
      theta0, initial theta(t=0)
      r0,     initial r(t=0) [AU]
      
    Returns:
      arrays of r, theta values and x, y values
    
    """
    
    # debug: track each theta, dtheta
    #current_thetas = np.zeros(len(t_array))
    #dthetas = [0.]
    debug = False

    dt = P / n        # duration of time step
    print(f"dt = {dt.to(u.hr)} for {n} steps")
    t_array = np.linspace(0, P*nP, n*nP)  # array of time steps from 0 to P

    # initialize array of longitudes for each time step
    thetas = np.zeros(len(t_array))
    thetas[0] = theta0

    # initialize array of radii
    r = np.zeros(len(t_array)) * u.AU
    r[0] = r0

    # initialize x, y arrays (as defined in Murray+Correia)
    x, y, z = np.zeros((3, len(t_array))) * u.AU
    x[0] = r0 * np.cos(theta0)
    y[0] = r0 * np.sin(theta0)
    z[0] = 0.

    # debug: track each theta, dtheta
    #current_thetas = np.zeros(len(t_array))
    #dthetas = [0.]
    debug = False
    
    # calculate r, theta at each time step
    for i in range(len(t_array)):
        if debug & (i % 1000 == 0):
            print(f"step {i} / {n*nP}")
        
        # get time
        t = t_array[i]
        # get current longitutde
        if debug: print(f"i: {i}, t: {t.decompose()}")
        current_theta = thetas[i] # thetas[np.where(t_array == t)[0][0]]  # current value of theta
        # current_thetas.append(current_theta) # debug only

        # calculate dtheta
        dtheta = ((np.sqrt(const.G * mstar)) * (1 + e * np.cos(current_theta))**2 * ( a * (1 - e**2))**(-3/2) * dt).decompose()
        if debug: 
            print(f"theta + dtheta = {current_theta} + {dtheta} = {current_theta + dtheta}")
        # dthetas.append(dtheta)  # debug only

        # add to get new longitude 
        new_theta = current_theta + dtheta.value
        # save the new theta longitude to the array
        if i+1 == len(t_array):
            pass
        else:
            thetas[i+1] = new_theta

        # because we know theta, we can get the position
        current_r = a * (1 - e**2) / (1 + e * np.cos(current_theta))
        r[i] = current_r
        x[i] = current_r * np.cos(current_theta)  # ellipse major axis for omega = 0
        y[i] = current_r * np.sin(current_theta)  # ellipse minor axis for omega = 0
        z[i] = 0.
        
    return t_array, r, thetas, x, y, z

def unpack_star(star, test=Fale):
    """
    Get parameters for running Kepler solver, R-M calculation for a given 
       star-planet system
    
    Inputs:
        star, DataFrame with [vrot_data] + [koi_data] + [Gdoc data]
    Returns:
        a bunch of parameters
    
    """
    
    if test:
        tag = "TEST"
        koimstar, koirstar, koiPstar = 1., 1., 20.
    
    else:
        tag = star['Star']
        koimstar, koirstar, koiPstar = star['Mass'], star['Mass'], star['Prot'] 

    
    # star
    mstar = koimstar * u.M_sun  # stellar mass
    rstar = koirstar * u.R_sun  # stellar radius
    Pstar = koiPstar * u.d # stellar rotation period
    phi_star = 0.*(np.pi/180) 
    
    # planet
    mp = koimp * u.M_earth # 1 * u.M_jupiter # planet mass
    rp = koirp * u.R_earth # 1 * u.R_jup # planet radius


    

def main():
    """
    Run Rossiter-Mclaughlin analysis for a star-planet system.
    
    Inputs:
        star, DataFrame
    
    Returns:
        crime
    
    """
    
    ca
    
    kepler_solver()



if __name__ == "__main__":

    main()


