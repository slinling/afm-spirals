# afm-spirals
Analytical flyby framework (AFM) for spirals [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7156158.svg)](https://doi.org/10.5281/zenodo.7156158)

This is an analytical framework based on multi-variate normal distribuiton from *Gaia* DR3 outputs. See Shuai et al. (2022) for the mathematical derivation and corresponding paper (ApJS accepted).

### Step 0: Query the [Gaia DR3 archive](https://gea.esac.esa.int/archive/) using ADQL
Query the neighboring stars that are located within 10 pc from a given star. See [`Step0_ADQL.txt`](https://github.com/slinling/afm-spirals/blob/main/Step0_ADQL.txt) for an example using the MWC 758 system. Export the querying results in a `.csv` file in folder `./data_gaia_query/`.

### Step 1: Compute the frequentist results using the Gaia DR3 values
Run `python Step1_Frequentist.py SR21` at [Step1_Frequentist.py](https://github.com/slinling/afm-spirals/blob/main/Step1_Frequentist.py) for the SR21 system (no blank spaces because the code reads `sys.argv` parameters).

The code reads the `.csv` file exported from Step 0, then writes the frequentist results in a `.csv` file in folder `./data_fequentist/`.

### Step 2: Perform Bayesian calculation for closest approach *time* for selected flyby candidates
Run `python Step2_Bayesian.py SR21 68` at [Step2_Bayesian.py](https://github.com/slinling/afm-spirals/blob/main/Step2_Bayesian.py) for the SR21 system (no blank spaces because the code reads `sys.argv` parameters), where `68` is the row index of for SR21B in the corresponding `.csv` file from Step 1.

The code reads the `.csv` file exported from Step 1, then write the Bayesian results using [`emcee`](https://emcee.readthedocs.io/en/stable/) on closest-approach time in an `.h5` file in folder `./data_mcmc/`.

### Step 3: Perform Monte Carlo sampling for closest approach *distance* for selected flyby candidates
Run `python Step3_MC_distance.py SR21 68` at [Step3_MC_distance.py](https://github.com/slinling/afm-spirals/blob/main/Step3_MC_distance.py) to sample the distribution for closest approach distance. The distance samples will be stored at `./data_mcmc/` in a `.npy` file, see [distance_posterior_SR21.npy](https://github.com/slinling/afm-spirals/blob/main/data_mcmc/distance_posterior_SR21.npy) for the corresponding example for SR21 and SR21B.

## Citation
```
@software{shuai22,
  author       = {Linling Shuai and
                  Bin Ren},
  title        = {slinling/afm-spirals: pre-release v0.1},
  month        = oct,
  year         = 2022,
  publisher    = {Zenodo, doi: \href{https://doi.org/10.5281/zenodo.7156158}{10.5281/zenodo.7156158}},
  version      = {pre-release},
  doi          = {10.5281/zenodo.7156158},
  url          = {https://doi.org/10.5281/zenodo.7156158}
}
```
