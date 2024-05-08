# SFD: Sparse-Function-Decomposition-via-Orthogonal-Transformation

 Let $[d]:=\{1, \ldots, d\},$ $\mathbb B_r:= \{x \in \mathbb R^d: \|x\| \le r\},$ and $D:=\otimes_{i \in [d]} [a_i, b_i],$ such that $a_i < b_i \in \mathbb R$ for all $i \in [d]$. For a twice continuously differentiable function $f$, i.e. $f \in C^2(\mathbb B_r), f: \mathbb B_r \rightarrow \mathbb R,$ we aim to find an orthogonal matrix $U \in O(d)$ such that $f_U:= f(U\cdot)$ has a sparse additive structure, i.e.
    $$f_U(x) = \sum_{u \in S} \tilde f_u(x_u) , \quad x_u := (x_i)_{i \in u},$$
with $\mathcal{S} \subset P\left(\{1, \cdots, d\}\right), |\mathcal{S}|\ll 2^d$. The graph below below shows a bivariate function that can be decomposed as a sum of two univariate functions after a rotation of angle $\pi/4$ of the input space.

<p align="center">
<img src="https://github.com/fatima0111/Sparse-Function-Decomposition-via-Orthogonal-Transformation/blob/main/Plots/Bivariate_functions/all_f2.png" width="500" height="500">
</p>
<p align="center"> 
    <em>Top-left: $f(x)= \sin(5 u_1^{\tT}x) + \sin(5 u_2^{\tT}x)$ where $U=(u_1, u_2) \in SO(2)$  with rotation angle $\pi/4.$ Top-right: $\partial_{x_1}f(x).$ Bottom-left: $f_U(x):=f(Ux).$ Bottom-right: $\partial_{x_1}f_U(x)$ </em>
</p>
Two tipical techniques to find sparse additive decompositions of functions are given by the anchored and the Anova decomposition. The sparsity of those functions decompositions is equivalent to the fact that most of term of the mixed partial derivative of the target function are vanishing. of The input space of the underlying target function can be rotated such the resulting function 
has a sparse decomposition, i.e. most of the summands in .. vanish. If the underlying function $f \in C^d(\mathbb B(r))$ it can be shown that $f_u$ to the partial derivative $\partial_u f=0$ of the function. Later we formule some optimization problems on the partial derivatives in order to get the orthogonal matrix $U \in O(d),$ such that the new resulting function $f_U:=f(U\cdot)$ has a sparse decomposition. 
Numerical examples on random sampled matrices and later on functions shows the reliability of our method.
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
| Output           | Testscripts for run example from section 5 and appendix D                                           |  
| Run_Test_Cases   | Testscripts for run example from section 5 and appendix D                                           |  
| Utils            | Implementation of numerical examples from Section 6. of [[1]](#1)                                   |


## Legal Information & Credits

Copyright (c) 2024 [Fatima Antarou Ba](https://www.tu.berlin/imageanalysis/ueber-uns/team), [Oleh Melnyk](https://olehmelnyk.xyz/), [Christian Wald](https://www.tu.berlin/imageanalysis/ueber-uns/team), [Gabriele Steidl](https://page.math.tu-berlin.de/~steidl/)

This software was written by Fatima Antarou Ba and Christian Wald and Oleh Melnyk. It was developed at the Institute of Mathematics, TU Berlin. Fatima Antarou Ba acknowledges support by the German Research Foundation within the Bundesministerium für Bildung und Forshung within the Sale project. Christian Wald acknowledges funding by the DFG within the SFB “Tomography Across the Scales” (STE 571/19-1, project
number: 495365311).

SFD is free software. You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version. If not stated otherwise, this applies to all files contained in this package and its sub-directories.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
