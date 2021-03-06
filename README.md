This repository is the matlab implement of the paper [**Frame-Based Sparse Analysis and Synthesis Signal Representations and Parseval K-SVD**](https://ieeexplore.ieee.org/document/8712410).


### ParsevalKSVD

Parseval K-SVD is the new approach of the K-SVD, which is a dictionary learning technique.

The Parseval K-SVD algorithm solve the dicitoinary $\psi$ and sparse coefficient by following:

![formulation]( https://lh3.googleusercontent.com/pw/ACtC-3cJEwVIvFlQMmqNDqY1-g7K26g-NBqYk3g9CVbn_VcpHh7KJZ9E6dFuHawiKzrAf0krVa41p4k6Fgmm9o0yQvoJSfwRiZE2AXAxSXOB2Eq3J3ARg_V3oNm5T7Xsqx7RQk7_1ASmuyhm-cCzVXj6X5N8=w529-h205-no?authuser=0)



The matlab function  **ParsevalKSVD.m** is the implement of  the Parseval K-SVD algorithm.

### Syntax

```matlab
[Psi, Phi, X, Record] = ParsevalKSVD(Y, Psi0, Phi0, X0, maxIter, t, rho, IsRecord, ShowDetail)
```

### The main file
* **.ipynb**
  
  ``` 
  Parseval_KSVD_Test.ipynb
  ```

* **.mlx**

  ```
  Parseval_KSVD_Test_mlx.mlx
  ```

* **.m**

  ```
  Parseval_KSVD_Test.m
  ```

  