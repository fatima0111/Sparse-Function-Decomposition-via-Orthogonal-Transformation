# SFD: Sparse-Function-Decomposition-via-Orthogonal-Transformation

Let $[d]:=\{1, \ldots, d\}$, $\mathbb B_r:=\{x \in \mathbb{R}^d: \|x\|\le r\}$, and $D := \bigotimes_{i \in [d]} [a_i, b_i]$, where $a_i < b_i \in \mathbb{R}$ for all $i \in [d]$. Consider a twice continuously differentiable function $f \in C^2(\mathbb B_r)$, with $f:\mathbb B_r \to \mathbb R$. Our goal is to find an orthogonal matrix $U \in O(d)$ such that, under the change of variables $f_U := f(U \cdot)$, the resulting function admits a sparse additive structure of the form 
 $$f_U(x) = \sum_{u \in S} \tilde f_u(x_u) , \quad x_u := (x_i)_{i \in u},$$
where $\mathcal{S} \subset \mathcal{P}(\{1, \ldots, d\})$ and $|\mathcal{S}| \ll 2^d$.
The plots below illustrate a bivariate function that can be expressed as the sum of two univariate functions after rotating its input domain by an angle of $\pi/4$.

<p align="center">
  <img src="https://github.com/fatima0111/Sparse-Function-Decomposition-via-Orthogonal-Transformation/blob/main/Plots/Bivariate_functions/all_f2.png?raw=true" width="500" height="500">
</p>
<p align="center">
  <em>
    <strong>Top-left:</strong> f(x) = sin(5 u₁ᵀx) + sin(5 u₂ᵀx), where U = (u₁, u₂) ∈ SO(2), with rotation angle π/4.  
    <strong>Top-right:</strong> ∂x₁f(x).  
    <strong>Bottom-left:</strong> f<sub>U</sub>(x) := f(Ux).  
    <strong>Bottom-right:</strong> ∂x₁f<sub>U</sub>(x).
  </em>
</p>
Two common methods for obtaining sparse additive decompositions of multivariate functions are based on the anchored decomposition and the ANOVA decomposition. The sparsity of these decompositions is closely tied to the vanishing of most mixed partial derivatives. In this work, we focus specifically on minimizing first- and second-order interactions that is, the first- and second-order mixed partial derivatives.

To achieve this, we define a graph associated with the target function, where the nodes correspond to nonzero first-order partial derivatives, and the edges represent nonzero second-order mixed partial derivatives.

Finding an orthogonal matrix that minimizes these first- and second-order interactions or equivalently, reduces the number of nodes and edges in the associated graph is accomplished through the following three steps:

   1. Vertex Reduction: Apply singular value decomposition (SVD) to decrease the number of vertices in the function graph.

   2. Block Diagonalization: Perform joint block diagonalization on a family of matrices to reveal underlying structural groupings.

   3. Edge Minimization: Use sparse optimization techniques (based on relaxations of the ℓ₀ “norm”) to reduce the number of edges.

For the final step, we develop and analyze optimization algorithms on the manifold of special orthogonal matrices (i.e., rotations). The figure below summarizes this three-step method.
 
<p align="center">
<img src="https://github.com/fatima0111/Sparse-Function-Decomposition-via-Orthogonal-Transformation/blob/main/Plots/3_step_optimization.png">
</p>
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

| **File/Folder**     | **Purpose**                                                                                                                    |
|---------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `Dataset`           | JSON files containing the datasets used in the numerical examples (Section 5 and Appendix D).                                  |
| `Evaluate_results`  | Scripts to evaluate the results of the numerical experiments in Section 5 and Appendix D, and to generate the corresponding plots. |
| `Libs`              | Implementation of the grid search, simultaneous block diagonalization, and manifold optimization methods described in [1].     |
| `Plots`             | Contains the generated plots presented in Section 5.                                                                           |
| `Run_Test_Cases`    | Test scripts for reproducing the examples from Section 5 and Appendix D.                                                       |
| `Utils`             | Helper functions for generating synthetic test data and supporting the main experiments.                                       |

## Legal Information & Credits

Copyright (c) 2024 [Fatima Antarou Ba](https://www.tu.berlin/imageanalysis/ueber-uns/team), [Oleh Melnyk](https://olehmelnyk.xyz/), [Christian Wald](https://www.tu.berlin/imageanalysis/ueber-uns/team), [Gabriele Steidl](https://page.math.tu-berlin.de/~steidl/)

This software was written by Fatima Antarou Ba and Christian Wald and Oleh Melnyk. It was developed at the Institute of Mathematics, TU Berlin. Fatima Antarou Ba acknowledges support by the German Research Foundation within the Bundesministerium für Bildung und Forshung within the Sale project. Christian Wald acknowledges funding by the DFG within the SFB “Tomography Across the Scales” (STE 571/19-1, project
number: 495365311).

SFD is free software. You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version. If not stated otherwise, this applies to all files contained in this package and its sub-directories.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
