Code base for the paper.

"A convex approach for learning near-isometric linear embeddings"
C. Hegde, A. Sankaranarayanan, W. Yin, R. Baraniuk
IEEE Transactions on Signal Processing, 63(22), pp.  6109-6121, 2015

@article{hegde2015numax,
  title={{NuMax: A} Convex Approach for Learning Near-Isometric Linear Embeddings},
  author={Hegde, Chinmay and Sankaranarayanan, Aswin C. and Yin, Wotao and Baraniuk, Richard},
  journal={IEEE Trans. Signal Processing},
  volume={63},
  number = {22},
  pages={6109--6121},
  year={2015},
  url_Paper={files/paper/2015/numax_tsp.pdf},
  pubtype = {Journal},
}

MAIN FUNCTIONS

1) NuMax.m: Solves the NuMax problem. Use this to solve vanilla version of NuMax with small sized problems. Number of secants < 10000 or so.

2) NuMax_CG.m: Main workhorse for any practical sized problem. Solves NuMax-CG problem for a dataset. This actively searches for "support" secants that satisfy the infinity norm constraints at boundary OR violate them and optimizes selectively over them. This code can be used for 100s-1000s of millions of secants or equivalents datasets of 10s of thousands of points. Fairly efficient implementation. Parameter tuning might be required to balance memory / computational efficiency.

Typical running time on the MNIST dataset where each data point is a 28x28 image.
40 mins for 4000 datapoints
2 hrs for 15000 datapoints
9 hrs for 60000 datapoints

These times reduce considerably on low dimensional data (i,e downsampled data).

3) NuMax_CG_Directional.m: NuMax-CG tailored for classification problem, Given class labels, it performs a limited form of RIP. upperbound of RIP is relaxed for point from different classes. Lowebound of RIP is relaxed for points from the same class.

4) demo_numax.m: A wrapper for NuMax solver. Runs a small sized problem over secants from a translating square manifold.

5) demo_numax_CG.m: A wrapper for NuMax-CG solver. Runs over a medium sized problem from MNIST database.

MINOR FUNCTION

1) funA_secants_WY.m, funAT_secants_WY.m: Implementation of the main linear operator and its adjoint. 

2) minFunc folder: a L-BFGS solver from http://www.di.ens.fr/~mschmidt/Software/minFunc.html
CGS is an alternative option. 
See 

3) get_rip_constants: given data matrix and projection operator, it finds the worst case RIP constants

4) LinearTimeSVD -- an approximate SVD solver. we use it for large dimensional problems. not tested extensively
