# Debiasing matrix completion via first estimating missingness probabilities via matrix completion

This code reproduces results from our NeurIPS 2019 paper "Missing Not at Random in Matrix Completion: The Effectiveness of Estimating Missingness Probabilities under a Low Nuclear Norm Assumption".

Authors: George H. Chen (georgechen@cmu.edu), Wei Ma (wei.w.ma@polyu.edu.hk)

We have also included code by some other authors, namely:

- ExpoMF (Liang et al 2016): https://github.com/dawenl/expo-mf
- A Python implemention by github user `andrewdalex` of SoftImpute-ALS (Hastie et al 2015): https://github.com/andrewdalex/SoftImpute-ALS

# Code dependencies

We tested this code using Anaconda Python 3.7 in a Linux environment (Ubuntu 18.04) with these additional packages:

- surprise (install using `pip install -U surprise`)
- copt (install using `pip install -U copt`)
- hnswlib (install using `pip install hnswlib`)

We modified Surprise's SVD and SVDpp to allow for weighted entries, and we also have helper functions coded in cython; these require cython compilation:

```
python setup_mnar_mc_helpers.py build_ext --inplace
```

# Getting the code to run

To run the code, you must first prepare datasets and then you can run `python demo.py <dataset name>` (edit `demo.py` to specify which matrix completion algorithms to run).

For the synthetic datasets, prepare them by running `python prepare_synthetic.py`.

Then you should be able to run `python demo.py steck-0` (as well as `steck-1`, `steck-2`, up through `steck-9` for the MovieLoverData and `useritemfeature-0`, `useritemfeature-1`, up through `useritemfeature-9` for the UserItemData).

For the Coat dataset, download it here: https://www.cs.cornell.edu/~schnabts/mnar/

Copy it to `./coat/`

Run `python prepare_coat.py`. Then you should be able to run `python demo.py coat`.

For the MovieLens-100k dataset, download it here: https://grouplens.org/datasets/movielens/100k/

Copy it to `./ml-100k/`

Run `python prepare_ml100k.py ml-100k-0` (as well as `ml-100k-1`, `ml-100k-2`, up through `ml-100k-9`)
