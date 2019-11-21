function [P_max,H]=EntropyRateOp(A,PI)
% EntropyRateOp(A,PI) is a function to calculate the probability ransition
% matrix with maximum entropy rate corraletd to give graph and stationary
% distribution.
% 
% The mathematical form can be found in 
% <https://ieeexplore.ieee.org/abstract/document/8371596>
% Example
%   A=[1 1 0;1 1 1;0 1 1];
%   PI=[1/6;1/2;1/3];
%   [P_op,H]=EntropyRateOp(A,PI);
n=size(A,2);
v_place=[];
for i=1:n
    for j=1:n
        v_place=[v_place j+(i-1)*n];
    end
end
A_inv=ones(n,n);
A_inv(A>0)=0;
A_vec=zeros(n^2,1);
A_inv_vec=zeros(n^2,1);
A_eq1=zeros(n,n^2);
for i=1:n^2
    A_vec(i)=A(i);
    if A_inv(i)==1
        A_inv_vec(i)=A_inv(i);
        A_eq1=[A_eq1;A_inv_vec'];
        A_inv_vec=zeros(n^2,1);
        continue
    end       
end
    function f=myfun(x,PI)
        P=zeros(n,n);
        P(v_place)=x;
        PI=repmat(PI,[1,n]);
        P(P==0)=1;
        f=-sum(sum(PI.*P.*log(P)));
        f=-real(f);
    end
x0=1/n*ones(n^2,1);
for i=1:n
    for j=i:n:n^2
        A_eq1(i,j)=1;
    end
end
b_eq1=[ones(n,1);zeros(size(A_eq1,1)-n,1)];
A1=[diag(A_vec);-diag(A_vec)];
b1=[ones(n^2,1);-zeros(n^2,1)];
a=1;
    function [c,ceq]=nonlcon1(x)
        c=[];
        P=zeros(n,n);
        P(v_place)=x;
        ceq=PI'*P-PI';
    end
        options = optimoptions('fmincon','Algorithm','sqp','MaxFunctionEvaluations',10000000,'MaxIterations',10000000);
        [y,f] = fmincon(@(x)myfun(x,PI),x0,A1,b1,A_eq1,b_eq1,[],[],@nonlcon1,options);
        P_max=zeros(n,n);
        P_max(v_place)=y;
        H=-f;
end



