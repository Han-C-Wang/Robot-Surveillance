module MixingTimeEvaMod
using Base,LinearAlgebra
export MixingTimeEva
"""
MixingTime(P) is a function used to calculate the SLEM of a probability
transition matrix. The mathematical form can be found in
<https://web.stanford.edu/~boyd/papers/pdf/fmmc.pdf>

The input parameter of the function is a symmetric probability transition matrix.

#Example
```test
  P=[1/3 1/2 1/6;1/5 2/5 2/5;1/7 2/7 4/7]
  f=MixingTime(P)
```
"""
function MixingTimeEva(P)
    n=size(P,2)
    P_one=P-ones(n,n)
    eigenvalue=eigen(P_one'*P_one)
    return maximum(eigenvalue.values)
end
end
