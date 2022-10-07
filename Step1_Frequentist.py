#!/usr/bin/env python
# coding: utf - 8

#### This function performs frequentist calculation to obtain closest
####     approach distance and time
#### 1. It calculates the values
#### 2. Then sorts the neighbors based on their closest distance to the host

# Inputs from command line:
#   sys.argv[1]: name for a spiral host
#   sys.argv[2]: dist_to_name parameter exported from Gaia query

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


def calculate_t ():
"""
Calculate the time and distance of the closest encounter and return the Gaia IDolklm
 that meets the constraints (time within past 10^4 years and distance within 10
 * radius of the disk)
"""
    Data  = pd.read_csv ( path_csv ) # Read Gaia query results
    Data1 = Data.loc[:, ['ra','dec','pmra','pmdec','pm','dist_arcsec',sys.argv[2],'z_pc','designation','ruwe']]
    
    # position information of all neighboring stars（now）
    DEC = np.array ( Data1['dec'] )
    cos_DEC = np.cos ( np.radians ( DEC ) )
    RA = np.array ( Data1['ra'] )
    ID = np.array ( Data1['designation'] )
    RUWE = np.array ( Data1['ruwe'] )

    # position and velocity information of the host star（now )
    x0 = Data1.iloc[0, 0]
    y0 = Data1.iloc[0, 1]
    cos_DEC0 = np.cos ( np.radians ( y0 ) )
    pmra0 = Data1.iloc[0, 2]
    pmdec0 = Data1.iloc[0, 3]
    z0 = Data1.iloc[0, 7]
    
    # relative position（now）
    unit_c = 3600 * 1000   # Unit conversion：deg. to  mas
    RA_r =  ( RA - x0 ) * unit_c  # unit: mas
    Dec_r =  ( DEC - y0 ) * unit_c

    # relative velocity（now）
    PMRA_r = np.array ( Data1['pmra']/cos_DEC - pmra0/cos_DEC0 )  # unit: mas/yr
    PMDEC_r = np.array ( Data1['pmdec'] - pmdec0 )
    
    # calculate the closest distance
    a = PMDEC_r
    b =  - PMRA_r
    c = PMRA_r * Dec_r - PMDEC_r * RA_r
    f_mas = abs ( c )/ (  ( a ** 2 + b ** 2 ) ** 0.5 ) # unit: mas
    f_deg = f_mas/unit_c
    d_pc = z0 * np.radians ( f_deg ) * au.pc
    d_au = d_pc.to ( au.au )             # unit: au
    
    # RA coordinates at the closest distance
    alpha =  ( - a * c )/ ( a ** 2 + b ** 2 ) # unit: mas
    tt =  -  (  ( RA_r - alpha )/PMRA_r ) # unit: year
    
    # limit the range of time t
    t_where = np.where ( tt  >  0 )
    t1 = np.piecewise ( tt, [tt > 0,tt < 0], [lambda tt:0,lambda tt:tt] )
    t2 = np.piecewise ( t1, [t1 <  - 1e + 04,t1 >  =  - 1e + 04], [lambda t1: - 1e + 04,lambda t1:t1] )
    
    Data0 = pd.DataFrame ( {'designation':ID,'dd/au':d_au,'t/yr':t2,'ra/deg':RA, 'dec/deg':DEC,'RUWE':RUWE} )
    
    return Data0, cos_DEC

# calculate the mean, z - value and standard deviation at the closest encounter
def calculate_z ( tt,num ):

    data  = pd.read_csv ( path_csv )    # Read Gaia query results
    data1 = data.loc[[0], ['ra','ra_error','dec','dec_error','pmra','pmra_error','pmdec','pmdec_error','ra_dec_corr','ra_pmra_corr','ra_pmdec_corr','dec_pmra_corr','dec_pmdec_corr','pmra_pmdec_corr','z_pc']]
    data2 = data.loc[[num], ['ra','ra_error','dec','dec_error','pmra','pmra_error','pmdec','pmdec_error','ra_dec_corr','ra_pmra_corr','ra_pmdec_corr','dec_pmra_corr','dec_pmdec_corr','pmra_pmdec_corr','z_pc']]
    
    data11 = data1.to_numpy ()[0]
    data22 = data2.to_numpy ()[0]
    
    R_a = data11[14]
    
    unit_c = 3600 * 1000   # unit conversion：deg. to  mas
    
    alpha_a = data11[0] * unit_c
    delta_a = data11[2] * unit_c
    cos_delta_a = np.cos ( np.radians ( data11[2] ) )
    mu_alpha_a = data11[4]/cos_delta_a
    mu_delta_a = data11[6]

    alpha_b = data22[0] * unit_c
    delta_b = data22[2] * unit_c
    cos_delta_b = np.cos ( np.radians ( data22[2] ) )
    mu_alpha_b = data22[4]/ cos_delta_b
    mu_delta_b = data22[6]
    
    sigma_alpha_a = data11[1]/cos_delta_a
    sigma_delta_a = data11[3]
    sigma_mu_alpha_a = data11[5]/cos_delta_a
    sigma_mu_delta_a = data11[7]
    
    sigma_alpha_b = data22[1]/cos_delta_b
    sigma_delta_b = data22[3]
    sigma_mu_alpha_b = data22[5]/cos_delta_b
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
    X_a = np.array ( [alpha_a,delta_a,mu_alpha_a,mu_delta_a] )
    X_b = np.array ( [alpha_b,delta_b,mu_alpha_b,mu_delta_b] )
    X_a = X_a.reshape ( 4,1 )
    X_b = X_b.reshape ( 4,1 )
    
    Sigma_a = np.array ( [[sigma_alpha_a ** 2,rho_a12 * sigma_alpha_a * sigma_delta_a,rho_a13 * sigma_alpha_a * sigma_mu_alpha_a,rho_a14 * sigma_alpha_a * sigma_mu_delta_a],
              [rho_a12 * sigma_alpha_a * sigma_delta_a,sigma_delta_a ** 2,rho_a23 * sigma_delta_a * sigma_mu_alpha_a,rho_a24 * sigma_delta_a * sigma_mu_delta_a],
              [rho_a13 * sigma_alpha_a * sigma_mu_alpha_a,rho_a23 * sigma_delta_a * sigma_mu_alpha_a,sigma_mu_alpha_a ** 2,rho_a34 * sigma_mu_alpha_a * sigma_mu_delta_a],
              [rho_a14 * sigma_alpha_a * sigma_mu_delta_a,rho_a24 * sigma_delta_a * sigma_mu_delta_a,rho_a34 * sigma_mu_alpha_a * sigma_mu_delta_a,sigma_mu_delta_a ** 2]] )

    Sigma_b = np.array ( [[sigma_alpha_b ** 2,rho_b12 * sigma_alpha_b * sigma_delta_b,rho_b13 * sigma_alpha_b * sigma_mu_alpha_b,rho_b14 * sigma_alpha_b * sigma_mu_delta_b],
                  [rho_b12 * sigma_alpha_b * sigma_delta_b,sigma_delta_b ** 2,rho_b23 * sigma_delta_b * sigma_mu_alpha_b,rho_b24 * sigma_delta_b * sigma_mu_delta_b],
                  [rho_b13 * sigma_alpha_b * sigma_mu_alpha_b,rho_b23 * sigma_delta_b * sigma_mu_alpha_b,sigma_mu_alpha_b ** 2,rho_b34 * sigma_mu_alpha_b * sigma_mu_delta_b],
                  [rho_b14 * sigma_alpha_b * sigma_mu_delta_b,rho_b24 * sigma_delta_b * sigma_mu_delta_b,rho_b34 * sigma_mu_alpha_b * sigma_mu_delta_b,sigma_mu_delta_b ** 2]] )
    
    # current position coordinates
    r_a = np.array ( [alpha_a,delta_a] )
    r_b = np.array ( [alpha_b,delta_b] )
    r_a = r_a.reshape ( 2,1 )
    r_b = r_b.reshape ( 2,1 )

    # position coordinates at time = t
    ct = np.array ( [[1,0,tt,0],[0,1,0,tt]] )
    mu_rt_a = ct@X_a
    mu_rt_b = ct@X_b

    # relative position coordinates at time = t
    mu_delta_rt = mu_rt_b - mu_rt_a
    Sigma_delta_rt = ct@Sigma_b@ct.T + ct@Sigma_a@ct.T

    mu_delta_RA = mu_delta_rt[0][0]
    mu_delta_Dec = mu_delta_rt[1][0]
    mu_delta_r =  (  ( mu_delta_RA ) ** 2 +  ( mu_delta_Dec ) ** 2 ) ** 0.5

    # rotation matrix
    sin_theta_t = mu_delta_Dec/mu_delta_r
    cos_theta_t = mu_delta_RA/mu_delta_r
    U = np.array ( [[cos_theta_t,sin_theta_t],[ - sin_theta_t,cos_theta_t]] )

    mu_t = U@mu_delta_rt # relative position after rotation
    Sigma_t = U@Sigma_delta_rt@U.T
    mu = mu_t[0][0] # relative distance after rotation
    
    # unit conversion: mas to au
    mu_deg = mu/unit_c
    mu_pc = R_a * np.radians ( mu_deg ) * au.pc
    mu_au = mu_pc.to ( au.au )
    
    # Standard Deviation  ( SD )
    sigma =  ( Sigma_t[0,0] ) ** 0.5
    
    # unit conversion: mas to au
    sigma_deg = sigma/unit_c
    sigma_pc = R_a * np.radians ( sigma_deg ) * au.pc
    sigma_au = sigma_pc.to ( au.au )
    
    z_au = mu_au/sigma_au            # unit: au

    return num,mu_au.value,z_au,sigma_au.value
    
# Calculate the time and distance of the closest encounter and return the number that that meets the constraints
DATA0,cos_delta = calculate_t ()
num_all = Data0.shape[0]

# calculate and output the mean, z - value and standard deviation at the closest encounter for each neiborning star
data1 = []
for i in range ( 0, num_all ):
        num_i = i
        tt = DATA0.at[i,'t/yr']
        j = calculate_z ( tt,num_i )
        data1.append ( j )
    
# merge two tables
DATA1 = pd.DataFrame ( data1,columns = ["index","d/au","z","d_error/au"] )
DATA0.insert ( 0, 'index', range ( 0, 0  +  len ( DATA0 ) ) )
outfile  =  pd.merge ( DATA1, DATA0 )

# sort by distance d
Data_d =  outfile.sort_values ( by = 'd/au', ascending = True )
Data_d = Data_d.drop ( columns = ["index"] )
Data_d = Data_d.drop ( columns = ["dd/au"] )

# reordering columns
order  =  ['designation', 'd/au', 'd_error/au','t/yr','z','ra/deg','dec/deg','RUWE']
data_d  =  Data_d[order]

data_d.to_csv ( path_data + sys.argv[1] + '.csv' )

sys.exit ( 0 )
