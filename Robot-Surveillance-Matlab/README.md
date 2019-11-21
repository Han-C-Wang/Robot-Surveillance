Robot-Surveillance-Matlab
======
This is a free open source matlab toolbox for calculating and optimizing a bunch of quantities and metrics that are related to Markov chains. It is motivated by the use of Markov chains in robotic applications where one or a group of robots randomly move on a graph to perform surveillance tasks. Details of the algorithms and mathematical formulas may be found in [professor Francesco Bullo's publications](http://motion.me.ucsb.edu/papers/Keyword/ROBOTIC-NETWORKS.html).
# Installation
* To install the toolbox into your matlab, please download [Robot Surveillance.mltbx](https://github.com/SJTUHan/Robot-Surveillance-Matlab/blob/master/Robot%20Surveillance.mltbx) into your workspace of matlab, then double click the file and all the functions in the toolbox can be used.
* The other way to use the toolbox is to download and unzip the software package, then add all files into the workspace. 
# Usage
## Notice
Functions in the package can be categorized into two types: evaluation functions and optimization functions. For convex optimization problems, [CVX](http://cvxr.com/cvx/) is used, which means that CVX is necessary when using optimization functions of this kind. For non-convex optimization problems, [fmincon](https://www.mathworks.com/help/optim/ug/fmincon.html) is used. The users may set the options of the fmincon function by locating the function in the files, and no interfaces for changing them are offered in this version.
## Example
Detailed user instructions can be found in documentations of the functions. See below for an example. 

```
A=[1 1 0;
   1 0 1;
   0 1 1];
W=[1 2 3;
   4 5 6;
   7 8 9];
tau=10;
[F,K]=MC_OP(P,W,tau,'HittingTimeOp');
```

# Checking Function
|Function|Julia|Matlab|
|:-:|:-:|:-:|
|Legal Markov Chain|√|√|
|Irreducible|√|√|
|Legal Option|√|√|
|Default Weighted Matrix|√|√|
|Symmetric Matrix (some case)|√|√|
|Dimension Match|√|√|
|Legal Stationary Distribution|√|√|
|Integer and non-negative(duration)|√|√|
