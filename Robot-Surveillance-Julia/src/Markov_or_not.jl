"""
Markov_or_not(P) is used to determine whether the
input probability transition matrix is a legal markov chain
or not.

#Example
```test
  P=[1/2 1/3;1/2 1/2]
  Markov_or_not(P)
```
"""
function Markov_or_not(P)
n=size(P,2);
for i=1:n
    if abs(sum(P[i,:],dims=1)[1]-1)>=1e-5
       error("the matrix you input is an illegal probability transition matrix")
    end
end
end
