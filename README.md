
## Name
Fisher Forecast Code for CAMB called Bobby

## Description
Fisher Forecast Code for CAMB, validated against Ilich+(2022). This version is an on-going version with unfinished features. Use it with care.

The code is fine tuned for EFTCAMB in GR but commenting specific 2 lines in the grid_IST recovers CAMB.
Given the gird_IST.py code, it is easy to implement new parameters in the grid for the forecasts, simply by adding the parameters to the class of parameters and the variations in the function that calls CAMB/EFTCAMB.

The default derivative method is polynomial fitting. Alternatively, there's also finite-different method of computing derivatives, which is faster but gives a steeper dependency on ell for the derivatives of the galaxy clustering probes.In addition, there's a derivative calculated using the Richardson Extrapolation method. The three methods handles numerical errors in different ways. A general rule of thumb is that the more points of the grid, the higher the accuracy and the higher the agreement between methodologies.

Polynomial fitting is slower but in perfect agreement with Ilich+(2022) forecasts. The accuracy is described by the R^2 test (which quantifies the goodness of the fit) which is computed alongside the derivative.
Finite Difference method is faster but it is different. In Derivate_coeff.py there's the code that computes the coefficients for the differences and you can call those functions to compute higher-order derivatives too. In this case the error scales to the power of number of points considered

Richardson's Extrapolation is under testing due to weird behaviour for some specific derivatives. In general, this method tries to cancel out the leading term in the error when approximating a derivative to a numerical derivative.

Matrix inversion is set as numpy.linalg.inv for semplicity and accuracy. It exploits LU decomposition for the inversion. Higher accuracy methods are under testing (they seem to not add much more on the final forecasts)


## Installation
Install CAMB or EFTCAMB as usual, remember to check the paths in the code so they call camb/eftcamb.

## Usage
Tutorial grid showcases how to compute the grids of parameters which are the building blocks to compute derivatives and the Fisher.
Tutoria_fisher showcases the computation of a Fisher both using a all-in-one function and all the single functions.

## Support
If you find and bug or something that does not work, please email me at guglielmo.frittoli@roma2.infn.it

## Roadmap
- Implement new features (eg. higher precision inversion) and stabilize the ones that are there (e.g. quantify better the differences in numerical derivatives methods).
- Finish and validate the code for SN
- Finish and validate the code for Best Fit parameters shift and Bayes Factor Expected value (for modelling and model comparisons).
- The user will be provided with more options for convenience. I will make the code more user-friendly and accessible.

