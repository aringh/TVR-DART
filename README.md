TVR-DART and DART implemented with ODL and ASTRA
================================================

This repository contains the code for the article "High-level algorithm prototyping: an example
extending the TVR-DART algorithm" by A. Ringh, X. Zhuge, W. J. Palenstijn, K. J. Batenburg, and O. Öktem.

Contents
--------
The code contains the following

* Files containing the implementation of the relevant operators and functionals.
* A script running the TVR-DART algorithm. All figures in the article is from this script.
* Code from [odl](https://github.com/odlgroup/odl) which contains minor modifications compared to commit [32842320a](https://github.com/odlgroup/odl/commit/32842320ad5b91ef07da17a5dc6f9292318706df) (which is essentially release 0.6.0, when it comes to changes in these parts of the code). These are default_functionals.py, steplen.py, and tensor_ops.py. The modifications are marked in the code
* A script running the DART algorithm.
* A test phantom.

Installing and running the code
-------------------------------
Clone the repository. Then, using miniconda run the following commands to set up a new environment (essentially follow the [odl installation instructions](https://odlgroup.github.io/odl/getting_started/installing.html))
* $ conda create -c odlgroup -n my_env python=3.5 odl=0.6.0 matplotlib pytest scikit-image spyder
* $ source activate my_env
* $ conda install -c astra-toolbox astra-toolbox

After this, the scripts can be run using, e.g., spyder.

Contact
-------
[Axel Ringh](https://www.kth.se/profile/aringh), PhD student  
KTH Royal Institute of Technology, Stockholm, Sweden  
aringh@kth.se

[Xiaodong Zhuge](https://www.cwi.nl/people/2717), Researcher  
Centrum Wiskunde & Informatica, Amsterdam, The Netherlands  
x.zhuge@cwi.nl

[Willem Jan Palenstijn](https://www.cwi.nl/people/2649), Researcher  
Centrum Wiskunde & Informatica, Amsterdam, The Netherlands  
willem.jan.palenstijn@cwi.nl

[Joost Batenburg](https://www.cwi.nl/people/1742), Professor  
Centrum Wiskunde & Informatica, Amsterdam, The Netherlands  
Leiden University, Leiden, The Netherlands  
joost.batenburg@cwi.nl

[Ozan Öktem](https://www.kth.se/profile/ozan), Associate Professor  
KTH, Royal Institute of Technology  
ozan@kth.se


