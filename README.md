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
This repository collects examples for the usage of the LSST DESC Core Cosmology Library (`CCL`). `CCL` provides routines to compute basic cosmological observables with validated numerical accuracy. The library is written in C99 and all functionality is directly callable from C and C++ code.  We also provide python bindings for higher-level functions. The examples in this repository illustrate usage in python specifically. 

The material in this repository was developed within the LSST DESC using LSST DESC resources. DESC users should use it in accordance with the [LSST DESC publication policy](http://lsstdesc.org/Collaborators). External users are welcome to use the code outside DESC in accordance with the licensing information below.

# CCL documentation

Further CCL documentation can be found in our [Readthedocs page](https://readthedocs.org/projects/ccl/). 
More examples can be found in our benchmark comparison notebooks [here](https://github.com/LSSTDESC/CCL/tree/master/examples) and in deprecated (pre-CCL v2) notebooks [here](https://github.com/LSSTDESC/CCL/tree/v2.0.1/examples). The examples in this repository continue evolving to match developments in the CCL master branch, but you can find examples that are v2-compatible if you restrict to the first set of commits.

# License, Credits, Feedback etc
The `CCL` code has been released by DESC, although it is still under active development. It is accompanied by a journal paper that describes the development and validation of `CCL`, which you can find on the  arxiv:[1812.05995](https://arxiv.org/abs/1812.05995). If you make use of the ideas or software here, please cite that paper and provide a link to this repository: https://github.com/LSSTDESC/CCL. You are welcome to re-use the code, which is open source and available under terms consistent with our [LICENSE](https://github.com/LSSTDESC/CCL/blob/master/LICENSE), which is a [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause) license. 

For free use of the `CLASS` library, the `CLASS` developers require that the `CLASS` paper be cited: CLASS II: Approximation schemes, D. Blas, J. Lesgourgues, T. Tram, arXiv:1104.2933, JCAP 1107 (2011) 034. The `CLASS` repository can be found in http://class-code.net. Finally, CCL uses code from the [FFTLog](http://casa.colorado.edu/~ajsh/FFTLog/) package.  We have obtained permission from the FFTLog author to include modified versions of his source code.

# Contact
If you have comments, questions, or feedback, please [write us an issue](https://github.com/LSSTDESC/CCLX/issues). You can also contact the [administrators](https://github.com/LSSTDESC/CCL/CCL-administrators).

