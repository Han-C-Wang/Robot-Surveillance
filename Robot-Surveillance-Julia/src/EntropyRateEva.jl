module EntropyRateEvaMod
include("stadis.jl")
using Base,LinearAlgebra,.stadisMod
export EntropyRateEva
"""
EntropyRate(P) is a function used to
calculate the entropy rate of a
probability transition matrix.
The mathematical form can be found in
<https://ieeexplore.ieee.org/abstract/document/8371596>

Example
```test
julia> P=[1/3 1/2 1/6;1/5 2/5 2/5;1/7 2/7 4/7]
3×3 Array{Float64,2}:
 0.333333  0.5       0.166667
 0.2       0.4       0.4
 0.142857  0.285714  0.571429

julia> f=EntropyRateEva(P)
-1.003863718382082
 ```
"""
function EntropyRateEva(P)
PI=stadis(P)
n=size(P,2)
f=0;
for i=1:n
    for j=1:n
        if  abs(P[i,j])>1e-5
            f=f+PI[i]*P[i,j]*log(P[i,j])
        end
    end
end
return f
end
end
