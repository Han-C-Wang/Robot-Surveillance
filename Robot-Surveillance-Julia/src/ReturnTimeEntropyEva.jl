module ReturnTimeEntropyEvaMod
using Base,LinearAlgebra
include("stadis.jl")
include("HittingTimeEva.jl")
using .stadisMod,.HittingTimeEvaMod
export ReturnTimeEntropyEva
"""
ReturnTimeEntropy(P,W,eta) is a function to calculate
the return time entropy with predefined probability
transition matriix, weighted matrix, and truncated parameter.
The mathematical form can be found in <https://ieeexplore.ieee.org/abstract/document/8675541>

#Example
```test
  P=[1/3 1/2 1/6;1/5 2/5 2/5;1/7 2/7 4/7]
  W=[1 2 3;4 5 6;7 8 9]
  eta=0.01
  f=ReturnTimeEntropyEva(P,W,eta)
```
"""
function ReturnTimeEntropyEva(P,W,eta)
n=size(W,2);
PI=stadis(P)
w_max=maximum(maximum(W))
w_max_int=w_max[1,1]
PI_min=minimum(PI);
N_eta=ceil(w_max/(eta*PI_min[1,1]))-1;
J=0;
F=HittingTimeEva(P,W,convert(Int64,N_eta));
for k=1:convert(Int64,N_eta)
    for i=1:n
        if abs(F[k*n+i,i])<=1e-5
            Entropy=0;
        else
            Entropy=F[k*n+i,i]*log(F[k*n+i,i]);
        end
        J=J-PI[i]*Entropy;
    end
end
return J[1,1]
end
end
