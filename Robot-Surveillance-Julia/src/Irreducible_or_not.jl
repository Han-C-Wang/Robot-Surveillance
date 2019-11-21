"""
Irreducible_or_not(P) is used to determine
whether the input square matrix is irreducible
or not

#Example
```test
  P=[1 0 1;1 0 1;1 0 1]
  bool=Irreducible_or_not(P)
```
"""
function Irreducible_or_not(P)
n=size(P,2)
A=zeros(n,n)
for i=1:n
    for j=1:n
        if(P[i,j]>0)
            A[i,j]=1;
        end
    end
end
suma=zeros(n,n)
for i=1:n
    suma=suma+A
    A=A*A
end
if minimum(minimum(suma))==0
    error("the matrix you have input is reducible!")
end
end
