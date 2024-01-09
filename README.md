<!---
STYLE CONVENTION USED   
    bolt italic:
        ***file***"
    code:
       `program` or `library``
       `commands` or `paths`
       `variable`
    bold code:
        **`function`**
        **`type`** or **`structure`**
-->
# CCLX     
This repository collects examples for the usage of the LSST DESC [Core Cosmology Library](https://github.com/LSSTDESC/CCL/) (`CCL`) in python. 

The material in this repository was developed within the LSST DESC using LSST DESC resources. DESC users should use it in accordance with the [LSST DESC publication policy](http://lsstdesc.org/Collaborators). External users are welcome to use the code outside DESC in accordance with the licensing information below.

## Suggested flow
Although each notebook in this directly can be explored independently of all the others, here is a suggested flow if you are just getting started with `CCL`: The [Distance Calculations Example](https://github.com/LSSTDESC/CCLX/blob/master/Distance%20Calculations%20Example.ipynb) will lead you through the basics of setting up a cosmology object in `CCL` and using it from essential calculations based on the cosmological model, such as distances. At this level, you can also explore the [Power spectrum example](https://github.com/LSSTDESC/CCLX/blob/master/Power%20spectrum%20example.ipynb) which shows you several options for modelling the matter power spectrum. (Variants of these predictions are provided in the [PerturbationTheoryPk](https://github.com/LSSTDESC/CCLX/blob/master/PerturbationTheoryPk.ipynb) and the [Halo-model-Pk](https://github.com/LSSTDESC/CCLX/blob/master/Halo-model-Pk.ipynb)notebooks). 

`CCL` uses power spectra and distances to obtain angular power spectra for LSST observbles and their corresponding correlation functions (see [CellsCorrelations](https://github.com/LSSTDESC/CCLX/blob/master/CellsCorrelations.ipynb)). That notebook gives you the essential LSST angular power spectra for lensing and clustering, but you might want to use `CCL` for other types of correlations (see [GeneralizedTracers](https://github.com/LSSTDESC/CCLX/blob/master/GeneralizedTracers.ipynb)). An [MCMC noteboook](https://github.com/LSSTDESC/CCLX/blob/master/MCMC%20Likelihood%20Analysis.ipynb) illustrates how one can use `CCL` with a likelihood sampler to get cosmological constraints given a set of data.  

`CCL` is also capable of giving cluster number counts predictions, as in [Halo-mass-function-example](https://github.com/LSSTDESC/CCLX/blob/master/Halo-mass-function-example.ipynb). You might want to make predictions for other clusters observables. See the [Halo-profiles](https://github.com/LSSTDESC/CCLX/blob/master/Halo%20profiles.ipynb) notebook.

All of the above predictions work for wCDM models. For the beyond-wCDM cases, see the [MG_mu_sigma_examples](https://github.com/LSSTDESC/CCLX/blob/master/MG_mu_sigma_examples.ipynb).

Finally, for some utilities, such as reading and writing `CCL` cosmologies, see [this notebook](https://github.com/LSSTDESC/CCLX/blob/master/Reading-writing-Cosmology-objects.ipynb).

## Running on binder
These notebooks can be run online using [binder](http://mybinder.org/v2/gh/LSSTDESC/CCLX/master) (kudos to Antony Lewis for this). Note that some notebooks featuring functionality that hasn't yet been formally released may not run.

## Running on Google colab

It is possible to install and run these examples, or your own CCL code, from [Google Colab](https://colab.research.google.com/). To install CCL on Colab, you can do the following (thanks to Jean-Eric Campagne):

```
!pip install -q condacolab

import condacolab

condacolab.install()

!conda install --no-pin pyccl

import pyccl
```


# CCL documentation

Further CCL documentation can be found in our [Readthedocs page](https://readthedocs.org/projects/ccl/). 
More examples can be found in our benchmark comparison notebooks [here](https://github.com/LSSTDESC/CCL/tree/master/examples) and in deprecated (pre-CCL v2) notebooks [here](https://github.com/LSSTDESC/CCL/tree/v2.0.1/examples). The examples in this repository continue evolving to match developments in the CCL master branch, but you can find examples that are v2-compatible if you restrict to the first set of commits.

# License, Credits, Feedback etc
The `CCL` code has been released by DESC and is accompanied by a journal paper that describes the development and validation of `CCL v1.0.0`, which you can find on the  arxiv:[1812.05995](https://arxiv.org/abs/1812.05995). The latest version is [CCL v2.0.1](https://github.com/LSSTDESC/CCL/releases/tag/v2.0.1). 

If you make use of the ideas or software here, please cite the `CCL` [paper](https://ui.adsabs.harvard.edu/abs/2019ApJS..242....2C/abstract) and provide a link to this repository: https://github.com/LSSTDESC/CCL. You are welcome to re-use the code, including these examples, which is open source and available under terms consistent with our [LICENSE](https://github.com/LSSTDESC/CCL/blob/master/LICENSE), which is a [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause) license. 

For free use of the `CLASS` library, the `CLASS` developers require that the `CLASS` paper be cited: CLASS II: Approximation schemes, D. Blas, J. Lesgourgues, T. Tram, arXiv:1104.2933, JCAP 1107 (2011) 034. The `CLASS` repository can be found in http://class-code.net. Finally, CCL uses code from the [FFTLog](http://casa.colorado.edu/~ajsh/FFTLog/) package.  We have obtained permission from the FFTLog author to include modified versions of his source code.

# Contact
If you have comments, questions, or feedback, please [write us an issue](https://github.com/LSSTDESC/CCLX/issues). You can also contact the [administrators](https://github.com/LSSTDESC/CCL/CCL-administrators).

