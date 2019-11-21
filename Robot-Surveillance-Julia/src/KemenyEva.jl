module KemenyEvaMod
include("stadis.jl")
using Base,LinearAlgebra,.stadisMod
export KemenyEva
"""
Kemeny(P,W) is a function used to calculate
the kemeny constant with predefined probability
transition matrix and weighted matrix. The
mathematical form can be found in <https://ieeexplore.ieee.org/abstract/document/7094271>

#Example
```test
  P=[1/3 1/2 1/6;1/5 2/5 2/5;1/7 2/7 4/7]
  W=[1 2 3;4 5 6;7 8 9]
  K=KemenyEva(P,W)
```
"""
function KemenyEva(P,W)
PI=stadis(P)
F=eigen(P)
K=1
for i=1:size(P,2)
    if abs(F.values[i]-1)>1e-5
        K=K+1/(1-F.values[i])
    end
end
n=size(P,2)
K=PI'*(P.*W)*ones(n,1)*K;
return K[1,1]
end
end
