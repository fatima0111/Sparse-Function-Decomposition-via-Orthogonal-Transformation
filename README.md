# Sparse-Function-Decomposition-via-Orthogonal-Transformation

Decomposition of any higher dimensional smooth function into a sum of lower dimensional function, i.e.  
f(x) = sum.....
Two tipical examples are given by the Anchored and the Anova decomposition. The input space of the underlying target function can be rotation such the resulting function 
has a sparse decomposition, i.e. most of the summands in .. vanish. In  our work we relate each term f_u to the partial derivative of the function. Later we formule some optimization problems on the partial derivatives in order to get the orthogonal matrix such that the new function has a sparse decomposition. 
Numerical examples on random sampled matrices and later on functions shows the reliability of the algorithm.


## Prerequisites
To run the code, the following packages have to be installed

 [Numpy](https://numpy.org/citing-numpy/)
 
 [Pytorch](https://pytorch.org/) (Cuda version)
 
 [Tntorch](https://tntorch.readthedocs.io/en/latest/) (to compute the ANOVA decomposition of the test functions)
 
 [Geoopt](https://geoopt.readthedocs.io/en/latest/manifolds.html) (to run RiemannianSGD for the rectraction method)
 
 [LandingSGD](https://github.com/pierreablin/landing) (to run the Landing method)
 
#### Make sure that the fastsum package is in the same parent directory as the NFFT-Sinkhorn directories.


## Reference

When you are using this code, please cite the paper

<a id="1">[1]</a> Fatima Antarou Ba, Michael Quellmalz. **Accelerating the Sinkhorn algorithm for sparse multi-marginal optimal transport via fast Fourier transforms**. 
_Algorithms_ 2022, 15(9), 311; [doi:10.3390/a15090311](https://doi.org/10.3390/a15090311) (Open Access)

This paper also explains the algorithms in more detail.

## Directory structure

| File/Folder   | Purpose                                                                                   |
| ------------- |-------------------------------------------------------------------------------------------|   
| Libs          | Sinkhorn Algorithm from Section 4., and NFFT-Sinkhorn algorithm from Section 5. of [[1]](#1) |
| images        | Marginal images for Wasserstein barycenters with general tree                                 |
| Output_Circle | Output of the numerical examples for MOT problem with tree-structured cost function       |
| Output_Tree   | Output of the numerical examples for MOT problem with tree-structured cost function       |
| Test_functions| Implementation of numerical examples from Section 6. of [[1]](#1)                           |
| Utils         | Auxiliary methods for the Sinkhorn algorithm and the numerical examples                 | 


## Legal Information & Credits

Copyright (c) 2024 [Fatima Antarou Ba](https://www.tu.berlin/imageanalysis/ueber-uns/team), [Oleh Melnyk](https://olehmelnyk.xyz/), [Christian Wald](https://www.tu.berlin/imageanalysis/ueber-uns/team), [Gabriele Steidl](https://page.math.tu-berlin.de/~steidl/)

This software was written by Fatima Antarou Ba and Christian Wald and Oleh Melnyk. It was developed at the Institute of Mathematics, TU Berlin. The first mentioned author acknowledges support by the German Research Foundation within the Bundesministerium für Bildung und Forshung within the Sale project.

SFD is free software. You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version. If not stated otherwise, this applies to all files contained in this package and its sub-directories.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
