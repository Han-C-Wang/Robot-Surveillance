module stadisMod
using Base,LinearAlgebra
export stadis
"""
stadis(P) is used to calculate the stationary distribution of
the input probability transition matrix.

#Example
```test
  P=[1/2 1/3 1/6;1/7 2/7 3/7;1/5 2/5 2/5]
  PI=stadis(P)
```
"""
function stadis(P)
F=eigen(copy(transpose(P)))
tmp=1
for i=1:size(P,2)
    if abs(F.values[i]-1)<1e-5
        tmp=i
        break
    end
end
return PI=real(F.vectors[:,tmp]./sum(F.vectors[:,tmp]))
end
end
