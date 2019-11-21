Robot-Surveillance-julia
======
This is a free open source matlab toolbox for helping with calculating and optimizing some stochastic parameters on markov chain. It is basicly used in robot application when one or a group of robots randomly move on a graph to perform a surveillance task. Details of the algorithms and mathematic forms can be found in [professor Francesco Bullo's publications](http://motion.me.ucsb.edu/papers/index.html).
# Installation
* Step 1: Add the package into your workspace
```julia
Pkg.add("https://github.com/SJTUHan/Robot-Surveillance-julia.git")
```
* Step 2: Use the package
```julia
using MarkovChain
```
# Usage
## Notice
Functions in the package can be separated into two types: evaluation and optimization. For convex optimization problems, [Convex.jl](https://github.com/JuliaOpt/Convex.jl) is used. For non-convex problems, [JuMP](https://github.com/JuliaOpt/JuMP.jl) is used. It should be noted that users can change the solver of JuMP in the functions by themselves, no additional interfaces for changing them are offered in this version.
## Example
Detailed user instructions can be found in documentations of the functions. See below for an example. 

```julia
A=[1 1 0;
   1 0 1;
   0 1 1]
W=[1 2 3;
   4 5 6;
   7 8 9]
tau=10
[F,K]=MC_OP(P,W,tau,"HittingTimeOp")
```
# Comparison with Matlab
Comparation results are shown in this [page](https://github.com/SJTUHan/Robot-Surveillance-Matlab/blob/master/README.md)
