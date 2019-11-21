module MixingTimeOpMod
export MixingTimeOp
using Base,LinearAlgebra,Convex,SCS
"""
MixingTimeOp(A) is a function to calculate the probability transition
matrix with maximum SLEM correlated to the given grah adjcent matrix.

In this function we use CVX to solve a sqp form problem. Usage instruction
of CVX can be found in <https://github.com/JuliaOpt/Convex.jl#convexjl>

The mathematical form can be found in <https://web.stanford.edu/~boyd/papers/pdf/fmmc.pdf>

#Example
```test
  A=[1 1 1 0;1 0 1 1;1 1 1 1;0 1 1 0]
  P_op=MixingTimeOp(A)
```

"""
function MixingTimeOp(A)
n=size(A,2)
A_inv=ones(n,n);
for i=1:n
    for j=1:n
        if A[i,j]>0
            A_inv[i,j]=0
        end
    end
end
s=Variable()
P=Variable(n,n)
constraint1=(P-1/n*ones(n,1)*ones(1,n)+s*Matrix{Float64}(I, n, n) in :SDP)
constraint2=(-P+1/n*ones(n,1)*ones(1,n)+s*Matrix{Float64}(I, n, n) in :SDP)
constraint3=(P>=0)
constraint4=(P*ones(n,1)==ones(n,1))
constraint5=(P==P')
constraint6=(P.*A_inv==0)
problem = minimize(s)
problem.constraints += [constraint1, constraint2, constraint3,constraint4,constraint5,constraint6]
solve!(problem, SCSSolver(verbose=1))
return P.value
end
end
