#!/usr/bin/env python
# coding: utf - 8

#### This function performs frequentist calculation to obtain closest
####     approach distance and time
#### 1. It calculates the values
#### 2. Then sorts the neighbors based on their closest distance to the host

# Inputs from command line:
#   sys.argv[1]: name for a spiral host(no blank space)
# For example
#   python3 Step1_Frequentist.py MWC758

import sys
import os
import pandas as pd
import numpy as np
import astropy.units as au
import matplotlib.pyplot as plt
import csv

#np.seterr( divide = 'ignore', invalid = 'ignore')

path_top = './data_gaia_query/'                     # path to the data folder
path_csv = path_top + sys.argv[1] + '-result.csv'   # path to each data file
path_data = './data_fequentist/'                    # path to store data


def calculate_t():
    """
        Frequentist calculation of the time and distance of the closest encounter,
        returning a table with
                1. time (within within past 10^4 years) and
                2. the closest distance( without time constraints )
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
    tt =  - ( ( RA_r - alpha )/PMRA_r )         # unit: year
    
    # limit the range of time in past 10^4 years
    
    tt[tt >0] = 0
    tt[tt < -1e4] = -1e4
#    t1 = np.piecewise( tt, [tt > 0, tt < 0], [lambda tt:0, lambda tt:tt] )       # when t>0, replace by 0, otherwise the same
#    t2 = np.piecewise( t1, [t1 <  - 1e + 04, t1 > = - 1e + 04], [lambda t1: - 1e + 04, lambda t1:t1] )
    t2 = tt
    Data0 = pd.DataFrame( {'designation':ID, 'dd/au':d_au, 't/yr':t2, 'ra/deg':RA, 'dec/deg':DEC, 'RUWE':RUWE} )
    
    return Data0


def calculate_z( tt, num ):
    """
        calculate the mean, standard error(SE) and z-value(i.e. mean/SE) between the host star and one of the neighboring star at the closest encounter
    """
    data  = pd.read_csv( path_csv )    # Read Gaia query results
    
    # Focus on the indices that we are interested in:
    
    data1 = data.loc[[0], ['ra', 'ra_error', 'dec', 'dec_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'ra_dec_corr', 'ra_pmra_corr', 'ra_pmdec_corr', 'dec_pmra_corr', 'dec_pmdec_corr', 'pmra_pmdec_corr', 'z_pc']]     # host star information
    data2 = data.loc[[num], ['ra', 'ra_error', 'dec', 'dec_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'ra_dec_corr', 'ra_pmra_corr', 'ra_pmdec_corr', 'dec_pmra_corr', 'dec_pmdec_corr', 'pmra_pmdec_corr', 'z_pc']]      # neighboring information
    
    data11 = data1.to_numpy()[0]
    data22 = data2.to_numpy()[0]
    
    R_a = data11[14]        # the distance to the host star (unit: pc)
    
    unit_c = 3600 * 1000    # unit conversion：deg. to  mas
    
    # the location and proper motion of the host star
    alpha_a = data11[0] * unit_c                        # Right Ascension (unit:mas)
    delta_a = data11[2] * unit_c                        # Declination (unit:mas)
    cos_delta_a = np.cos( np.radians( data11[2] ) )     # Cosine value of the declination
    mu_alpha_a = data11[4]/cos_delta_a                  # Proper Motion in Right Ascension direction
    mu_delta_a = data11[6]                              # Proper Motion in Declination diretion
    
    # the location and proper motion of the neigboring star
    alpha_b = data22[0] * unit_c                        # Right Ascension (unit:mas)
    delta_b = data22[2] * unit_c                        # Declination (unit:mas)
    cos_delta_b = np.cos( np.radians( data22[2] ) )     # Cosine value of the declination
    mu_alpha_b = data22[4]/ cos_delta_b                 # Proper Motion in Right Ascension direction
    mu_delta_b = data22[6]                              # Proper Motion in Declination diretion
    
    # standard error in the location and proper motion of the host star
    sigma_alpha_a = data11[1]/cos_delta_a               # Standard error of right ascension
    sigma_delta_a = data11[3]                           # Standard error of declination
    sigma_mu_alpha_a = data11[5]/cos_delta_a            # Standard error of proper motion in right ascension direction
    sigma_mu_delta_a = data11[7]                        # Standard error of proper motion in declination direction
    
    # standard error in the location and proper motion of the neigboring star
    sigma_alpha_b = data22[1]/cos_delta_b               # Standard error of right ascension
    sigma_delta_b = data22[3]                           # Standard error of declination
    sigma_mu_alpha_b = data22[5]/cos_delta_b            # Standard error of proper motion in right ascension direction
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

    # transformation matrix at time = t, see Equation (B2)
    ct = np.array( [[1, 0, tt, 0], [0, 1, 0, tt]] )
    
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
    
    # unit conversion: mas to (deg to pc to) au
    mu_deg = mu/unit_c
    mu_pc = R_a * np.radians( mu_deg ) * au.pc
    mu_au = mu_pc.to( au.au )
    
    # standard error after rotation
    sigma = ( Sigma_t[0, 0] ) ** 0.5
    
    # unit conversion: mas to (deg to pc to) au
    sigma_deg = sigma/unit_c
    sigma_pc = R_a * np.radians( sigma_deg ) * au.pc
    sigma_au = sigma_pc.to( au.au )
    
    z_au = mu_au/sigma_au            # unit: au

    #return these values
    # num: the row number subtract 2 in Gaia query result csv file
    # mu_au: the distance of the closest encounter
    # z_au: the z-value(i.e. mean/standard error) of the closest encounter
    # sigma_au: the standard error of the closest encounter
    return num, mu_au.value, z_au, sigma_au.value
    
# Calculate the time and distance of the closest encounter and return a table with time (within within past 10^4 years) and the closest distance ( without time constraints; used for housekeeping, do not use this one. Correct ones will be calculated later.)
DATA0 = calculate_t()

# total number of stars of Gaia query results
num_all = DATA0.shape[0]

data1 = []
# calculate and save the mean, standard error(SE) and z-value(i.e. mean/SE) for all neighborning stars
for i in range( num_all ):
    tt = DATA0.at[i, 't/yr']     # read in the time of closest encounter(within past 10^4 years) of each neighborning star
    info_ca = calculate_z( tt, i )    # calculate the mean, standard error(SE) and z-value(i.e. mean/SE) for each neighborning star
    data1.append( info_ca )
    if i % 100 == 0:
        print("Finished calculation for ", i, ' of ', num_all, ' neighbors.')
    
# merge two tables
DATA1 = pd.DataFrame( data1, columns = ["index", "d/au", "z", "d_error/au"] )
DATA0.insert( 0, 'index', range( 0, 0  +  len( DATA0 ) ) )
outfile  =  pd.merge( DATA1, DATA0 )

# sort by distance d
Data_d =  outfile.sort_values( by = 'd/au', ascending = True )
Data_d = Data_d.drop( columns = ["index"] )
#Data_d = Data_d.drop( columns = [""] )
Data_d = Data_d.drop( columns = ["dd/au"] ) #drop unconstrained distance from DATA0 (i.e., values from DATA1 are kept)
#Data_d = Data_d.drop( columns = ["nan"] )

# reordering columns
order  =  ['designation', 'd/au', 'd_error/au', 't/yr', 'z', 'ra/deg', 'dec/deg', 'RUWE']
data_d  =  Data_d[order]

data_d.to_csv( path_data + sys.argv[1] + '.csv' )

print('Closest flyby info saved to ', path_data + sys.argv[1] + '.csv' )

sys.exit( 0 )
