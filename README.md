# SFD: Sparse-Function-Decomposition-via-Orthogonal-Transformation

Decomposition of any higher dimensional smooth function into a sum of lower dimensional function, i.e.  
$$f(x) = \sum_{u \in S} f_u(x_u), \qquad x_u := (x_i)_{i \in u}$$
Two tipical examples are given by the Anchored and the Anova decomposition. The input space of the underlying target function can be rotated such the resulting function 
has a sparse decomposition, i.e. most of the summands in .. vanish. In  our work we relate each term f_u to the partial derivative of the function. Later we formule some optimization problems on the partial derivatives in order to get the orthogonal matrix such that the new function has a sparse decomposition. 
Numerical examples on random sampled matrices and later on functions shows the reliability of the algorithm.

<p align="center">
<img src="https://github.com/fatima0111/Sparse-Function-Decomposition-via-Orthogonal-Transformation/blob/main/Plots/Bivariate_functions/all_f2.png" width="500" height="500">
</p>
<p align="center"> 
    <em>Computing fixed support image barycenter with NFFT-Sinkhorn algorithm where the four corner images are given </em>
</p>

## Prerequisites
To run the code, the following packages have to be installed

 [Numpy](https://numpy.org/citing-numpy/)
 
 [Pytorch](https://pytorch.org/) (Cuda version)
 
 [Tntorch](https://tntorch.readthedocs.io/en/latest/) (to compute the ANOVA decomposition of the test functions)
 
 [Geoopt](https://geoopt.readthedocs.io/en/latest/manifolds.html) (to run RiemannianSGD for the rectraction method)
 
 [LandingSGD](https://github.com/pierreablin/landing) (to run the Landing method)
 

## Reference

When you are using this code, please cite the paper

<a id="1">[1]</a> Fatima Antarou Ba, Oleh Melnyk, Christian Wald, Gabriele Steidl. **Sparse additive function decompositions facing basis transforms**. 
2024; [arXiv:2403.15563](https://arxiv.org/abs/2403.15563) 

This paper also explains the algorithms in more detail.

## Directory structure

| File/Folder      | Purpose                                                                                             |
| -------------    |-----------------------------------------------------------------------------------------------------|   
| Dataset          | Json-files containing Dataset used in the numerical examples (section 5 and appendix D)             |
| Generation       | Scripts to generate generate dataset (random matrices and functions) and plots of the paper         |
| Libs             | Implementation of the gid-search method and the simultaneously block-diagonalisation of the hessians|
| Plots            | contains the generated plots                                                                        |
| Run_Test_Cases   | Testscripts for run example from section 5 and appendix D                                           |  
| Utils            | Implementation of numerical examples from Section 6. of [[1]](#1)                                   |


## Legal Information & Credits

Copyright (c) 2024 [Fatima Antarou Ba](https://www.tu.berlin/imageanalysis/ueber-uns/team), [Oleh Melnyk](https://olehmelnyk.xyz/), [Christian Wald](https://www.tu.berlin/imageanalysis/ueber-uns/team), [Gabriele Steidl](https://page.math.tu-berlin.de/~steidl/)

This software was written by Fatima Antarou Ba and Christian Wald and Oleh Melnyk. It was developed at the Institute of Mathematics, TU Berlin. Fatima Antarou Ba acknowledges support by the German Research Foundation within the Bundesministerium für Bildung und Forshung within the Sale project. Christian Wald acknowledges funding by the DFG within the SFB “Tomography Across the Scales” (STE 571/19-1, project
number: 495365311).

SFD is free software. You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version. If not stated otherwise, this applies to all files contained in this package and its sub-directories.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
