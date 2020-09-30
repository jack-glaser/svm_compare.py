# svm_compare.py
In nonlinear support vector machine classification, observations are recast to a higher-dimensional domain before their margin is calculated. To perform this recasting, one must select a kernel function to perform the high-dimensional inner products (this is the primary parameter selection). This package provides an easy comparison for the performance of different kernel functions in binary SVM classification, implementing the analysis using an iterated quadratic program solver (quadprog). The package tests out-of-sample performance by splitting input data into a training and a test dataset. 

# Versioning 
`svm_compare.py` is compatible with python 3

# Usage
`svm_compare.py` is most easily applied directly from the command line. Format your data in a .csv file (as in the sample data provided in `[./data](./data)`), and in the terminal simply call: 
```
python main.py <path_to_dataset>
```

Training and testing accuracy will be printed in the command line.

# Kernels

Several commonly used pre-written kernals are included in `svm.py`. These include a linear kernel, a [polynomial kernel](https://en.wikipedia.org/wiki/Polynomial_kernel) with user-programmable mean and degree, and a [radial basis function kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) with user a programmable gamma parameter. User-written kernels can be added to `svm.py`. 
