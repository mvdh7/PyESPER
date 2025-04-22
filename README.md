# PyESPER

"Note:"
This is a preliminary set of code, for which the accompanying manuscript is under review. The accompanyint Python package is being created and will be released soon.

## Introduction
PyESPER is a Python implementation of MATLAB Empirical Seawater Property Estimation Routines ([ESPERs](https://github.com/BRCScienceProducts/ESPER), and the present version consists of a preliminary Jupyter Notebook which implements these routines. These routines provide estimates of seawater biogeochemical properties at user-provided sets of coordinates, depth, and available biogeochemical properties. Three algorithm options are available through these routines: 

1. Locally interpolated regressions (LIRs)
2. Neural networks (NNs)
3. Mixed

## Basic Use
For the present version, you will need to download the [PyESPER_Final.ipynb](https://github.com/LarissaMDias/PyESPER/blob/main/PyESPER_Final.ipynb) along with the affiliated neural network files within the [NeuralNetworks](https://github.com/LarissaMDias/PyESPER/tree/main/NeuralNetworks) folder, and the [SimpleCantEstimateLR_full.csv](https://github.com/LarissaMDias/PyESPER/blob/main/SimpleCantEstimateLR_full.csv) file for estimates involving anthropogenic carbon calculations (pH and dissolved inorganic carbon). 
