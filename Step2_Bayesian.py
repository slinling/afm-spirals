#!/usr/bin/env python
# coding: utf-8

#### This function performs bayesian approach to obtain closest approach time
#### 1. It calculates the values
#### 2. Then obtains the posterior distribution for the closest approach time by MCMC

# Inputs from command line:
#   sys.argv[1]: name for a spiral host
#   sys.argv[2]: radius of the protoplanetary disk


import sys
import os
import pandas as pd
import numpy as np
import astropy.units as au

from multiprocessing import Pool
import time
import emcee
import h5py

import multiprocessing
multiprocessing.set_start_method( "fork" )          # needed for Linling's laptop
    
path_top = './data_gaia_query/'                     # path to the data folder
path_csv = path_top + sys.argv[1] + '-result.csv'   # path to each data file
path_data = './data_mcmc/'                    # path to store data

t_traceback = 1e4                               # how many years to track back to


def calculate_t():
    """
        Frequentist calculation of the time and distance of the closest encounter,
        returning the row number in Gaia query result .csv file and the closest encounter time
    """
    Data  = pd.read_csv(path_csv) # Read Gaia query results
    
    # Focus on the indices that we are interested in:
    Data1 = Data.loc[ : , ['ra', 'dec', 'pmra', 'pmdec', 'pm', 'dist_arcsec', 'dist_to_host', 'z_pc', 'designation', 'ruwe']]
    
    # location and proper motion information of the host star（Gaia DR3 epoch)
    x0 = Data1.iloc[0, 0]                   # Right Ascension
    y0 = Data1.iloc[0, 1]                   # Declination
    cos_DEC0 = np.cos( np.radians( y0 ) )   # Cosine value of the declination
    pmra0 = Data1.iloc[0, 2]                # Proper Motion in Right Ascension direction times cosine value of the declination
    pmdec0 = Data1.iloc[0, 3]               # Proper Motion in Declination diretion
    z0 = Data1.iloc[0, 7]                   # the parallax of the host star (unit: pc)
    
    # location and proper motion information of all stars
    DEC = np.array( Data1['dec'] )          # Declination
    cos_DEC = np.cos( np.radians( DEC ) )   # Cosine value of the declination
    RA = np.array( Data1['ra'] )            # Right Ascension
    ID = np.array( Data1['designation'] )   # Gaia ID of all stars
    RUWE = np.array( Data1['ruwe'] )        # the renormalized unit weight error of all stars

    # relative Right Ascension and Declination between the host star and all neighboring stars（Gaia DR3 epoch）
    unit_c = 3600 * 1000                    # Unit conversion：deg. to  mas
    RA_r = ( RA - x0 ) * unit_c             # unit: mas
    Dec_r = ( DEC - y0 ) * unit_c           # unit: mas

    # relative proper motion between the host star and all neighboring stars（Gaia DR3 epoch）
    PMRA_r = np.array( Data1['pmra']/cos_DEC - pmra0/cos_DEC0 )  # relative Proper Motion in Right Ascension direction (unit: mas/yr)
    PMDEC_r = np.array( Data1['pmdec'] - pmdec0 )                # relative Proper Motion in Declination direction     (unit: mas/yr)
    
    # calculate the closest approach distance
    # refer to Equation (B10) in paper for a, b, and c
    a = PMDEC_r
    b =  - PMRA_r
    c = PMRA_r * Dec_r - PMDEC_r * RA_r
    
    f_mas = abs( c )/( ( a ** 2 + b ** 2 ) ** 0.5 )   # The distance of the closest approach (unit: mas)
    f_deg = f_mas/unit_c                              # Unit conversion：mas to deg
    d_pc = z0 * np.radians( f_deg ) * au.pc           # Unit conversion：deg to pc
    d_au = d_pc.to( au.au )                           # unit: au
    
    # Right Ascension of neighboring stars at the closest distance
    alpha = ( - a * c )/( a ** 2 + b ** 2 )     # unit: mas
    
    # the time of the closest encounter
    # refer to Equation (B12) in paper for tt
    t =  - ( ( RA_r - alpha )/PMRA_r )         # unit: year
    
    # the radius of protoplanetary disk
    r =  float( sys.argv[2] ) * au.AU
    
    Data0 = pd.DataFrame( {'designation':ID,'dd/au':d_au,'t/yr':t,'ra/deg':RA, 'dec/deg':DEC,'RUWE':RUWE} )
    
    # limit the time t within past t_traceback years
    # and limit the distance d within 10 times protoplanetary disk radius
    Data00 = Data0[( t <= 0 )&( t > -t_traceback )&( d_au < 10 * r )]
    
    # the number of stars that meet the constraints
    num = Data00.index.tolist()
    
    #return these values
    # num[0]: the row number in Gaia query result .csv file that meets the constraints
    # t[num[0]]: closest encounter time that meets the constraints
   
    return num[0],t[num[0]]
    
# calculate flat priors for the flyby times when x is within past t_traceback years
def log_prior( x ):
    if  x <= 0 and x > -t_traceback:
        return 0.0
    return  -np.inf

# calculate the log-likelihood value when time = x
def log_likelihood( x ):
    sigma_2, chi_2 = calc_std2_chi2( x )
    ll =  - 0.5  * chi_2 - 0.5 * np.log( sigma_2 ) - 0.5 * np.log( 2 * np.pi )
    return ll

# calculate the log-posterior value when time = x
def log_probability( x ):
    ll = log_likelihood( x )
    lp  =  log_prior( x )
    if not np.isfinite( lp ):
        return  -np.inf
    return lp  +  ll

# Calculate and find stars whose closest encounter time within the last t_traceback years and closest encounter distance within 10 times the disk radius, then output the row number in Gaia query result .csv file and the closest encounter time
num, t_all  = calculate_t()

data = pd.read_csv( path_csv )    # Read Gaia query results

# Focus on the indices that we are interested in:
data1 = data.loc[[0], ['ra', 'ra_error', 'dec', 'dec_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'ra_dec_corr', 'ra_pmra_corr', 'ra_pmdec_corr', 'dec_pmra_corr', 'dec_pmdec_corr', 'pmra_pmdec_corr', 'z_pc']]     # host star information
data2 = data.loc[[num], ['ra', 'ra_error', 'dec', 'dec_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'ra_dec_corr', 'ra_pmra_corr', 'ra_pmdec_corr', 'dec_pmra_corr', 'dec_pmdec_corr', 'pmra_pmdec_corr', 'z_pc']]      # neighboring star information

data11 = data1.to_numpy()[0]
data22 = data2.to_numpy()[0]

unit_c = 3600 * 1000    # unit conversion：deg. to  mas

# the location and proper motion of the host star
alpha_a = data11[0] * unit_c                        # Right Ascension (unit:mas)
delta_a = data11[2] * unit_c                        # Declination (unit:mas)
cos_delta_a = np.cos( np.radians( data11[2] ) )     # Cosine value of the declination
mu_alpha_a = data11[4] / cos_delta_a                # Proper Motion in Right Ascension direction
mu_delta_a = data11[6]                              # Proper Motion in Declination diretion

# the location and proper motion of the neigboring star
alpha_b = data22[0] * unit_c                        # Right Ascension (unit:mas)
delta_b = data22[2] * unit_c                        # Declination (unit:mas)
cos_delta_b = np.cos( np.radians( data22[2] ) )     # Cosine value of the declination
mu_alpha_b = data22[4] / cos_delta_b                # Proper Motion in Right Ascension direction
mu_delta_b = data22[6]                              # Proper Motion in Declination diretion

# standard error in the location and proper motion of the host star
sigma_alpha_a = data11[1] / cos_delta_a             # Standard error of right ascension
sigma_delta_a = data11[3]                           # Standard error of declination
sigma_mu_alpha_a = data11[5] / cos_delta_a          # Standard error of proper motion in right ascension direction
sigma_mu_delta_a = data11[7]                        # Standard error of proper motion in declination direction

# standard error in the location and proper motion of the neigboring star
sigma_alpha_b = data22[1] / cos_delta_b             # Standard error of right ascension
sigma_delta_b = data22[3]                           # Standard error of declination
sigma_mu_alpha_b = data22[5] / cos_delta_b          # Standard error of proper motion in right ascension direction
sigma_mu_delta_b = data22[7]                        # Standard error of proper motion in declination direction

# correlation of the host star
rho_a12 = data11[8]                                 # Correlation between right ascension and declination
rho_a13 = data11[9]                                 # Correlation between right ascension and proper motion in right ascension
rho_a14 = data11[10]                                # Correlation between right ascension and proper motion in declination
rho_a23 = data11[11]                                # Correlation between declination and proper motion in right ascension
rho_a24 = data11[12]                                # Correlation between declination and proper motion in declination
rho_a34 = data11[13]                                # Correlation between proper motion in right ascension and proper motion in declination

# correlation of the neigboring star
rho_b12 = data22[8]                                 # Correlation between right ascension and declination
rho_b13 = data22[9]                                 # Correlation between right ascension and proper motion in right ascension
rho_b14 = data22[10]                                # Correlation between right ascension and proper motion in declination
rho_b23 = data22[11]                                # Correlation between declination and proper motion in right ascension
rho_b24 = data22[12]                                # Correlation between declination and proper motion in declination
rho_b34 = data22[13]                                # Correlation between proper motion in right ascension and proper motion in declination

# Generalized coordinate (location and proper motion vector) of two stars（Gaia DR3 epoch)
X_a = np.array( [alpha_a, delta_a, mu_alpha_a, mu_delta_a] )
X_b = np.array( [alpha_b, delta_b, mu_alpha_b, mu_delta_b] )
X_a = X_a.reshape( 4, 1 )
X_b = X_b.reshape( 4, 1 )

# covariance matrix of the host star
Sigma_a = np.array( [[sigma_alpha_a ** 2, rho_a12 * sigma_alpha_a * sigma_delta_a, rho_a13 * sigma_alpha_a * sigma_mu_alpha_a, rho_a14 * sigma_alpha_a * sigma_mu_delta_a],
          [rho_a12 * sigma_alpha_a * sigma_delta_a, sigma_delta_a ** 2, rho_a23 * sigma_delta_a * sigma_mu_alpha_a, rho_a24 * sigma_delta_a * sigma_mu_delta_a],
          [rho_a13 * sigma_alpha_a * sigma_mu_alpha_a, rho_a23 * sigma_delta_a * sigma_mu_alpha_a, sigma_mu_alpha_a ** 2, rho_a34 * sigma_mu_alpha_a * sigma_mu_delta_a],
          [rho_a14 * sigma_alpha_a * sigma_mu_delta_a, rho_a24 * sigma_delta_a * sigma_mu_delta_a, rho_a34 * sigma_mu_alpha_a * sigma_mu_delta_a, sigma_mu_delta_a ** 2]] )

# covariance matrix of the neigboring star
Sigma_b = np.array( [[sigma_alpha_b ** 2, rho_b12 * sigma_alpha_b * sigma_delta_b, rho_b13 * sigma_alpha_b * sigma_mu_alpha_b, rho_b14 * sigma_alpha_b * sigma_mu_delta_b],
              [rho_b12 * sigma_alpha_b * sigma_delta_b, sigma_delta_b ** 2, rho_b23 * sigma_delta_b * sigma_mu_alpha_b, rho_b24 * sigma_delta_b * sigma_mu_delta_b],
              [rho_b13 * sigma_alpha_b * sigma_mu_alpha_b, rho_b23 * sigma_delta_b * sigma_mu_alpha_b, sigma_mu_alpha_b ** 2, rho_b34 * sigma_mu_alpha_b * sigma_mu_delta_b],
              [rho_b14 * sigma_alpha_b * sigma_mu_delta_b, rho_b24 * sigma_delta_b * sigma_mu_delta_b, rho_b34 * sigma_mu_alpha_b * sigma_mu_delta_b, sigma_mu_delta_b ** 2]] )
              

r_a = np.array( [alpha_a, delta_a] )             # current position vector of the host star
r_b = np.array( [alpha_b, delta_b] )             # current position vector of the neigboring star
r_a = r_a.reshape( 2, 1 )
r_b = r_b.reshape( 2, 1 )

def calc_std2_chi2( tt ):
    """
        calculate and return the standard error and chi-squared value between the host star and one of the neighboring star at the closest encounter
    """
    # transformation matrix at time = t, see Equation (B2)
    if type( tt ) == np.ndarray:
        ct = np.array( [[1,0,tt[0],0],[0,1,0,tt[0]]] )
    else:
        ct = np.array( [[1,0,tt,0],[0,1,0,tt]] )

    # Matrix product, see Equation (B3)
    mu_rt_a = ct@X_a                                # position vector of the host star at time = t
    mu_rt_b = ct@X_b                                # position vector of the neigboring star at time = t

    # relative position vector and covariance matrix at time = t, see Equation (B4)
    mu_delta_rt = mu_rt_b - mu_rt_a
    Sigma_delta_rt = ct@(Sigma_b + Sigma_a)@ct.T

    # relative position at time = t
    mu_delta_RA = mu_delta_rt[0][0]                 #along RA direction
    mu_delta_Dec = mu_delta_rt[1][0]                #along Dec direction
    mu_delta_r = ( ( mu_delta_RA ) ** 2 + ( mu_delta_Dec ) ** 2 ) ** 0.5 #total

    # rotation matrix
    sin_theta_t = mu_delta_Dec/mu_delta_r
    cos_theta_t = mu_delta_RA/mu_delta_r
    U = np.array( [[cos_theta_t, sin_theta_t], [ - sin_theta_t, cos_theta_t]] )

    mu_t = U@mu_delta_rt                # relative location vector after rotation
    Sigma_t = U@Sigma_delta_rt@U.T      # covariance matrix after rotation
    
    # relative distance along the parallel direction after rotation, see Equation (B8)
    mu = mu_t[0][0]
    
    # variance along the parallel direction after rotation
    sigma2 = Sigma_t[0,0]
    
    # chi-squared value
    chi2 = ( mu ) ** 2 / sigma2
    
    return sigma2, chi2

t_all = np.array( [t_all] )

if t_all == np.array( [] ):
    print( "No flyby within 10 times disc radius in the past t_traceback years" )
else:
    # obtain the posterior distribution for the closest approach time by MCMC
    
    n_dim  =  len( t_all )               # number of variables
    n_walkers  =  int( 4 * n_dim )       # an even number(  >  =  2 * n_dim )
    step  =  5000                        # how many steps are expected for MCMC to runx0
    var_values_init  = float( t_all[0] )
    filename =  path_data + sys.argv[1] + '.h5'

    with Pool() as pool:
        start  =  time.time()

        if not os.path.exists( filename ):  #initial run, no backend file existed
            
            # the backend file is used to store the status
            backend  =  emcee.backends.HDFBackend( filename )

            # set an ensemble MCMC sampler
            sampler  =  emcee.EnsembleSampler( nwalkers  =  n_walkers, ndim  =  n_dim, log_prob_fn = log_probability, pool  =  pool, backend = backend )

            # randomize the initial time vector within plus or minus 1000 years
            values_2  =  np.random.uniform( var_values_init - 1000, var_values_init + 1000,size  =  [n_walkers, n_dim] )
            
            print( 'First run success' )
            
            # iterate sampler for n=5000 steps iterations and return the result
            sampler.run_mcmc( values_2, nsteps  =  step, progress = True )
            
        else:
        #load the data directly from a backend file, this is used when you want to pick up from a previous MCMC run
            
            backend  =  emcee.backends.HDFBackend( filename )

            sampler  =  emcee.EnsembleSampler( nwalkers  =  n_walkers, ndim  =  n_dim, log_prob_fn = log_probability, pool  =  pool, backend = backend )
            
            print( 'Second run success' )

            sampler.run_mcmc( None, nsteps  =  step, progress = True )
             
        end  =  time.time()
        serial_time  =  end  -  start

        print( "Mltiprocess took {0:.1f} seconds".format( serial_time ) )

sys.exit( 0 )
