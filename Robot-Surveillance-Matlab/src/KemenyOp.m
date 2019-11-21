function [P_op,f]=KemenyOp(A,PI,W)
% KemenyOp(A,PI,W) is a function used to calculate the probability
% transition matrix with maxmimum kemeny constant correlated to given graph
% and stationary distribution.
% 
% In this function we use fmincon to solve optimization problem
% 
% The mathematical form can be found in 
% <https://ieeexplore.ieee.org/abstract/document/7094271>
% 
% Example
%   A=[1 1 0;1 0 1;0 1 1];
%   PI=[1/6;1/2;1/3];
%   W=[1 2 0;3 0 4;0 5 6];
%   [P_op,f]=KemenyOp(A,PI,W);
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
diag_PI=diag(PI);
q=sqrt(PI);
    function f=Rushabh_eva(x,PI,W)
        P=zeros(n,n);
        P(v_place)=x;
        weighted=PI'*(P.*W)*ones(n,1);
        f=weighted*trace(inv(eye(n)-sqrt(diag_PI)*P*diag_PI^(-1/2)+q*q'));
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
    function [c,ceq]=nonlcon1(x)
        c=[];
        P=zeros(n,n);
        P(v_place)=x;
        ceq=P.*repmat(PI,[1,n])-P'.*(repmat(PI',[n,1]));
    end
        options = optimoptions('fmincon','Algorithm','sqp','MaxFunctionEvaluations',10000000,'MaxIterations',10000000);
        [y,f] = fmincon(@(x)Rushabh_eva(x,PI,W),x0,A1,b1,A_eq1,b_eq1,[],[],@nonlcon1,options);
        P_op=zeros(n,n);
        P_op(v_place)=y;
end