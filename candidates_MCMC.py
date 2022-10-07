#! / usr / bin / env python
# coding: utf - 8

# sys.argv[1]: name
# sys.argv[2]: dist_to_name
# sys.argv[3]: radius of the protoplanetary disk
# calculate the mean and uncertainty with MCMC
## Bayesian Approach

import sys
import os
import pandas as pd
import numpy as np
import astropy.units as au
import matplotlib.pyplot as plt

from multiprocessing import Pool
import time
import emcee
import h5py
import corner

import multiprocessing
multiprocessing.set_start_method( "fork" )
os.environ["OMP_NUM_THREADS"]  =  "1"
np.seterr( divide = 'ignore',invalid = 'ignore' )

path_top = ' / Users / shuailin / Desktop / data_dr / '            # path to the data folder
path_csv = path_top + sys.argv[1] + '-result.csv'                  # path to each data file
path_data = ' / Users / shuailin / Desktop / '                     # path to store data

# Calculate the time and distance of the closest encounter and return the number and time that that meets the constraints
def calculate_t():

    Data  = pd.read_csv( path_csv ) # Read Gaia query results
    Data1 = Data.loc[:, ['ra','dec','pmra','pmdec','pm','dist_arcsec',sys.argv[2],'z_pc','designation','ruwe']]
    
    # position information of all neighboring stars（now）
    DEC = np.array( Data1['dec'] )
    cos_DEC = np.cos( np.radians( DEC ) )
    RA = np.array( Data1['ra'] )
    ID = np.array( Data1['designation'] )
    RUWE = np.array( Data1['ruwe'] )

    # position and velocity information of the host star（now )
    x0 = Data1.iloc[0, 0]
    y0 = Data1.iloc[0, 1]
    cos_DEC0 = np.cos( np.radians( y0 ) )
    pmra0 = Data1.iloc[0, 2]
    pmdec0 = Data1.iloc[0, 3]
    z0 = Data1.iloc[0, 7]
    
    # relative position（now）
    unit_c = 3600 * 1000   # Unit conversion：deg. to  mas
    RA_r = ( RA - x0 ) * unit_c  # unit: mas
    Dec_r = ( DEC - y0 ) * unit_c

    # relative velocity（now）
    PMRA_r = np.array( Data1['pmra'] / cos_DEC - pmra0 / cos_DEC0 )  # unit: mas / yr
    PMDEC_r = np.array( Data1['pmdec'] - pmdec0 )
    
    # calculate the closest distance
    a = PMDEC_r
    b =  - PMRA_r
    c = PMRA_r * Dec_r - PMDEC_r * RA_r
    f_mas = abs( c ) / ( ( a ** 2 + b ** 2 ) ** 0.5 ) # unit: mas
    f_deg = f_mas / unit_c
    d_pc = z0 * np.radians( f_deg ) * au.pc
    d_au = d_pc.to( au.au )             # unit: au
    
    # RA coordinates at the closest distance
    alpha = (  - a * c ) / ( a ** 2 + b ** 2 ) # unit: mas
    t =  -( ( RA_r - alpha ) / PMRA_r ) # unit: year
    
    r =  float( sys.argv[3] ) * au.AU
    
    Data0 = pd.DataFrame( {'designation':ID,'dd / au':d_au,'t / yr':t,'ra / deg':RA, 'dec / deg':DEC,'RUWE':RUWE} )
    
    # limit the range of time t and distance d
    Data00 = Data0[( t <  = 0 )&( t >  - 10e + 4 )&( d_au < 10 * r )]
    num = Data00.index.tolist()
    
    return num[0],t[num[0]]
    

def log_prior( x ):
    if  x <  = 0 and x >  - 1e + 04:
        return 0.0
    return  - np.inf


def log_likelihood( x ):
    sigma_2,chi_2 = function( x )
    ll =  - 0.5  * chi_2 - 0.5 * np.log( sigma_2 ) - 0.5 * np.log( 2 * np.pi )
    return ll

def log_probability( x ):
    ll = log_likelihood( x )
    lp  =  log_prior( x )
    if not np.isfinite( lp ):
        return  - np.inf
    return lp  +  ll

num, t_all  = calculate_t()

data  = pd.read_csv( path_csv )
data1 = data.loc[[0], ['ra','ra_error','dec','dec_error','pmra','pmra_error','pmdec','pmdec_error','ra_dec_corr','ra_pmra_corr','ra_pmdec_corr','dec_pmra_corr','dec_pmdec_corr','pmra_pmdec_corr']]
data2 = data.loc[[num], ['ra','ra_error','dec','dec_error','pmra','pmra_error','pmdec','pmdec_error','ra_dec_corr','ra_pmra_corr','ra_pmdec_corr','dec_pmra_corr','dec_pmdec_corr','pmra_pmdec_corr']]


data11 = data1.to_numpy()[0]
data22 = data2.to_numpy()[0]

unit_c = 3600 * 1000   # unit conversion：deg. to  mas
alpha_a = data11[0] * unit_c
delta_a = data11[2] * unit_c
mu_alpha_a = data11[4]
mu_delta_a = data11[6]
alpha_b = data22[0] * unit_c
delta_b = data22[2] * unit_c
mu_alpha_b = data22[4]
mu_delta_b = data22[6]
sigma_alpha_a = data11[1]
sigma_delta_a = data11[3]
sigma_mu_alpha_a = data11[5]
sigma_mu_delta_a = data11[7]
sigma_alpha_b = data22[1]
sigma_delta_b = data22[3]
sigma_mu_alpha_b = data22[5]
sigma_mu_delta_b = data22[7]
rho_a12 = data11[8]
rho_a13 = data11[9]
rho_a14 = data11[10]
rho_a23 = data11[11]
rho_a24 = data11[12]
rho_a34 = data11[13]
rho_b12 = data22[8]
rho_b13 = data22[9]
rho_b14 = data22[10]
rho_b23 = data22[11]
rho_b24 = data22[12]
rho_b34 = data22[13]

# position and velocity coordinates of two stars（now )
X_a = np.array( [alpha_a,delta_a,mu_alpha_a,mu_delta_a] )
X_b = np.array( [alpha_b,delta_b,mu_alpha_b,mu_delta_b] )
X_a = X_a.reshape( 4,1 )
X_b = X_b.reshape( 4,1 )

Sigma_a = np.array( [[sigma_alpha_a ** 2,rho_a12 * sigma_alpha_a * sigma_delta_a,rho_a13 * sigma_alpha_a * sigma_mu_alpha_a,rho_a14 * sigma_alpha_a * sigma_mu_delta_a],
          [rho_a12 * sigma_alpha_a * sigma_delta_a,sigma_delta_a ** 2,rho_a23 * sigma_delta_a * sigma_mu_alpha_a,rho_a24 * sigma_delta_a * sigma_mu_delta_a],
          [rho_a13 * sigma_alpha_a * sigma_mu_alpha_a,rho_a23 * sigma_delta_a * sigma_mu_alpha_a,sigma_mu_alpha_a ** 2,rho_a34 * sigma_mu_alpha_a * sigma_mu_delta_a],
          [rho_a14 * sigma_alpha_a * sigma_mu_delta_a,rho_a24 * sigma_delta_a * sigma_mu_delta_a,rho_a34 * sigma_delta_a * sigma_mu_delta_a,sigma_mu_delta_a ** 2]] )

Sigma_b = np.array( [[sigma_alpha_b ** 2,rho_b12 * sigma_alpha_b * sigma_delta_b,rho_b13 * sigma_alpha_b * sigma_mu_alpha_b,rho_b14 * sigma_alpha_b * sigma_mu_delta_b],
              [rho_b12 * sigma_alpha_b * sigma_delta_b,sigma_delta_b ** 2,rho_b23 * sigma_delta_b * sigma_mu_alpha_b,rho_b24 * sigma_delta_b * sigma_mu_delta_b],
              [rho_b13 * sigma_alpha_b * sigma_mu_alpha_b,rho_b23 * sigma_delta_b * sigma_mu_alpha_b,sigma_mu_alpha_b ** 2,rho_b34 * sigma_mu_alpha_b * sigma_mu_delta_b],
              [rho_b14 * sigma_alpha_b * sigma_mu_delta_b,rho_b24 * sigma_delta_b * sigma_mu_delta_b,rho_b34 * sigma_delta_b * sigma_mu_delta_b,sigma_mu_delta_b ** 2]] )

# current position coordinates
r_a = np.array( [alpha_a,delta_a] )
r_b = np.array( [alpha_b,delta_b] )
r_a = r_a.reshape( 2,1 )
r_b = r_b.reshape( 2,1 )

# calculate the standard deviation and chi squared value at the closest encounter
def function( tt ):

    # position coordinates at time = t
    if type( tt ) =  = np.ndarray:
        ct = np.array( [[1,0,tt[0],0],[0,1,0,tt[0]]] )
    else:
        ct = np.array( [[1,0,tt,0],[0,1,0,tt]] )

    mu_rt_a = ct@X_a
    mu_rt_b = ct@X_b

    # relative coordinates at time = t
    mu_delta_rt = mu_rt_b - mu_rt_a
    Sigma_delta_rt = ct@Sigma_b@ct.T + ct@Sigma_a@ct.T

    mu_delta_RA = mu_delta_rt[0][0]
    mu_delta_Dec = mu_delta_rt[1][0]
    mu_delta_r = ( ( mu_delta_RA ) ** 2 +( mu_delta_Dec ) ** 2 ) ** 0.5

    # rotation matrix
    sin_theta_t = mu_delta_Dec / mu_delta_r
    cos_theta_t = mu_delta_RA / mu_delta_r
    U = np.array( [[cos_theta_t,sin_theta_t],[ - sin_theta_t,cos_theta_t]] )

    mu_t = U@mu_delta_rt             # relative position after rotation
    Sigma_t = U@Sigma_delta_rt@U.T   # Standard Deviation squared
    mu = mu_t[0][0]                  # relative distance after rotation
    sigma2 = Sigma_t[0,0]
    chi2 = ( mu ) ** 2 / sigma2
    
    return sigma2,chi2

t_all = np.array( [t_all] )

 # MCMC
if t_all[0] =  = 'nan':
    print( "No flyby" )
    
else:
    n_dim  =  len( t_all )    # number of variables
    n_walkers  =  int( 4 * n_dim )     # an even number(  >  =  2 * n_dim )
    step  =  5000            # how many steps are expected for MCMC to runx0
    var_values_init  = float( t_all[0] )
    filename =  path_data + sys.argv[1] + '.h5'


    with Pool() as pool:
        start  =  time.time()

        if not os.path.exists( filename ):  #initial run, no backend file existed
            
            backend  =  emcee.backends.HDFBackend( filename )   # the backend file is used to store the status

            sampler  =  emcee.EnsembleSampler( nwalkers  =  n_walkers, ndim  =  n_dim, log_prob_fn = log_probability, pool  =  pool, backend = backend )

            values_2  =  np.random.uniform( var_values_init - 1000, var_values_init + 1000,size  =  [n_walkers, n_dim] )
            print( 'First run success' )
            
            sampler.run_mcmc( values_2, nsteps  =  step, progress = True )
            
        else:#load the data directly from a backend file, this is used when you want to pick up from a previous MCMC run
            
            backend  =  emcee.backends.HDFBackend( filename )

            sampler  =  emcee.EnsembleSampler( nwalkers  =  n_walkers, ndim  =  n_dim, log_prob_fn = log_probability, pool  =  pool, backend = backend )
            
            print( 'Second run success' )

            sampler.run_mcmc( None, nsteps  =  step, progress = True )
             
        end  =  time.time()
        serial_time  =  end  -  start

        print( "1 nodes  *  4 cores with multiprocess took {0:.1f} seconds".format( serial_time ) )

sys.exit( 0 )
