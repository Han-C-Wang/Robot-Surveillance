%there exists a similar function names "isreducible()" in 
%Econometrics Toolbox 
function Irreducible_or_not(P)
n=size(P,2);
A=P;
A(A>0)=1;
A_tmp=A;
sum=0;
for i=1:n
    sum=sum+A_tmp;
    A_tmp=A_tmp*A;
end
if min(min(sum))==0
    error('the matrix you have input is reducible!');
end
end
