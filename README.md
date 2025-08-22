# Fast and Simple Multiclass Data Segmentation: An Eigendecomposition and  Projection-Free Approach
C. Faccio, M. Porcelli, F. Rinaldi, M. Stoll

## Overview

Graph-based machine learning has seen an increased interest over the last decade with many connections to other fields of applied mathematics.
Learning based on partial differential equations, such as the phase-field Allen-Cahn equation, allows efficient handling of semi-supervised learning approaches on graphs. 

The numerical solution of the graph Allen-Cahn equation via a convexity splitting or the Merriman-Bence-Osher (MBO) scheme, albeit being a widely used approach, requires the calculation of a graph Laplacian eigendecomposition and repeated projections over the unit simplex to maintain valid partitions. The computational efficiency of those methods is hence severely limited by those two bottlenecks in practice, especially when dealing with large-scale instances.

In order to overcome these limitations, we propose a new framework combining a novel penalty-based reformulation of the segmentation problem, which ensures valid partitions (i.e., binary solutions) for appropriate parameter choices, with an eigendecomposition and projection-free optimization scheme, which has a small per-iteration complexity (by relying primarily on sparse matrix-vector products)
and guarantees good convergence properties. 

The task considered in this paper follows a semi-supervised learning of the node labels. This means we are given the labels of a potentially small subset of the nodes and want learn the labels of the remaining nodes while simultaneously leveraging labeled and unlabeled nodes.


## A new penalty-based model for multiclass data segmentation

The objective function for the semi-supervised learning method we propose is derived from the Ginzburg–Landau energy functional on graphs with a new penalty term for the multi-class case

$$ E(U) = \frac{1}{2} trace(U^T L_s U) + \frac{1}{\epsilon} trace(U(II-U^T)) +
             \frac{1}{2} trace((\hat{U}-U)^T D_{\omega} (\hat{U}-U)) \quad \quad \quad (1)$$

where:
- $U = (u_1, . . . , u_n)^T \in \mathbb{R}^{n×K}$ where the k-th component of $u_i \in \mathbb{R}^K$ is the strength for data point $i$ to belong to class $k$ (n is the number of data points and K the number of classes);
- $\hat{U} = (\hat{u}_1, . . . , \hat{u}_n)^T \in\mathbb{R}^{n×K}$ where the k-th component of $\hat{u}_i \in \mathbb{R}^K$ is the known value of the fidelity node;
- $L_s \in \mathbb{R}^{n×n}$ is the graph Laplacian;
- $D_{\omega} = diag(\omega_1, ..., \omega_n) \in \mathbb{R}^{n×n}$ is a diagonal matrix with elements $\omega_i$ associated to supervised data;
- $\epsilon \in \mathbb{R}^+$ is the penalization parameter;
- $II \in \mathbb{R}^{K×n}$ is a matrix of all ones.

Therefore, the optimization problem we have is 

$$\min_{U \in \Sigma} E(U) $$

whit $\Sigma$ is the cartesian products of unit simplices.

## Solvers

In this repository four algorithms are implemented to address the multi-class problem (1):
 1. Convex splitting method, CS [4],
 2. Merriman-Bence-Osher scheme, MBO[2],
 3. classic Frank-Wolfe method, FW [1],
 4. our greedy version of FW, GFW [1].

#### The Greedy Frank-Wolfe (GFW) method

1. Choose a point $U_0 \in \Sigma$ 
2. For $k = 0,\dots $
3. ...... Compute $S_k \in GLMO_{\Sigma}(\nabla E(U_k))$
4. ...... If $S_k$ satisfies some specific condition, then STOP 
5. ...... Set $D_k = S_k − U_k$ 
6. ...... Set $U_{k+1} = U_k + \alpha_k D_k$, with $\alpha_k \in (0, 1]$ a suitably chosen stepsize
7. End for

We define the Greedy Linear Minimization Oracle **GLMO**  as follows: 

$$ GLMO_{\Sigma}(G)=(w_{1},..., w_{n})^T $$  

with 
- $G=(g_1,\dots,g_n)^T=\nabla E(U)$
- $w_i = e_{j_i}$, if $i \in I$ = { $i \in [1:n]: u_i \notin$  {0,1} $^K$ }, $j_i \in argmin_{j \in supp(u_i)} (c_i)_j$,
- $w_i = u_i$ otherwise.

Here, $supp(u_i)$ denotes the indices of the nonzero elements of the vector $u_i$. More specifically, in the GLMO we solve for each i such that $u_i \notin$ {0,1} $^K$ the following problem:

$$    \min_{y \in \bar\Delta^i_{K-1}} g_i^T y$$

with $\bar\Delta^i_{K-1}$ :={ $y \in \mathbb{R}^K_+ :\mathbf{1}^T y=1, y_j=0\ \forall\ j\notin supp(u_i)$}, while we do nothing and choose $u_i$ in case $u_i \in$ {0,1} $^K$.

## Prerequisites

Before using this code, you have to download the NFFT3 toolbox [5] from [here](https://www-user.tu-chemnitz.de/~potts/nfft/) and run the `configure` script with option `--with-matlab=/PATH/TO/MATLAB/HOME/FOLDER`.
Afterwards run `make` and `make check`. When calling this function, the folder `%NFFT-MAIN%/matlab/fastsum` must be in your MATLAB path.

The following MATLAB Toolboxes are required:
- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox

## Folders/files description

| Folder/file  | Purpose |
| ------------- | ------------- |
| auxils | contains the functions needed to run the main codes |
| lihi | contains functions used for subroutines creating sparse graph laplacians  |
| results | results will be saved here (text files and images) |
| solvers  | contains the solvers CS, MBO and FW (two versions are available, the classic and the greedy one) |
| testsets | contains the data sets needed to run the codes |
| README.md | this file  |
| run_images.m | main file to run CS, MBO and GFW method for image labelling. Four testing images are supplied: beach, 3 and 4 geometric figures and sheets of papers |
| run_networks.m | main file to run CS, MBO and GFW method for the segmentation of real networks: LFR benchmark data set, Twitch, LastFM, Facebook, Amazon (computers and photos) |
| run_synthetic_test.m | main file to run CS, MBO and GFW method for the segmentation of syntethic data sets |


## References
 
 [1] C. Faccio, M. Porcelli, F. Rinaldi, M. Stoll, "Fast and Simple Multiclass Data Segmentation: An Eigendecomposition and  Projection-Free Approach", pp. 1-21, arXiv:2508.09738.
 
 [2] C. Garcia-Cardona, E. Merkurjev, A. L. Bertozzi, A. Flenner, and A. G. Percus, "Multiclass data segmentation using diffuse interface methods on graphs", IEEE Trans. Pattern Anal. Mach. Intell., 36 (2014), pp. 1600–1613,
 
 [3] D. Alfke, D. Potts, M. Stoll, and T. Volkmer, "NFFT meets Krylov methods: Fast matrix-vector products for the graph Laplacian of fully connected networks", Frontiers in Applied Mathematics and Statistics, 4 (2018), p. 61.

 [4] A. L. Bertozzi, S. Esedoglu, and A. Gillette, "Inpainting of binary images using the Cahn–Hilliard equation", IEEE Transactions on image processing, 16 (2006), pp. 285–291

 [5] D. Alfke, D. Potts, M. Stoll, and T. Volkmer, "NFFT meets Krylov methods: Fast matrix-vector products for the graph Laplacian of fully connected networks", Frontiers in Applied Mathematics and Statistics, 4 (2018), p. 61.


## Conditions of use
Use at your own risk! No guarantee of any kind given or implied.

Copyright (c) 2025, C. Faccio, M. Porcelli, F. Rinaldi, M. Stoll. All rights reserved.

Redistribution and use of the provided codes in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

## Disclaimer 
This software is provided by the copyright holders and contributors "as is" and any express or implied warranties, including, but nor limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the copyright owner or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.  
