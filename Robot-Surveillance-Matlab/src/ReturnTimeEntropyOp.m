function [P_op,f]=ReturnTimeEntropyOp(A,PI,W,epsilon,eta)
% ReturnTimeEntropyOp(A,PI,W,epsilon,eta) is a function to calculate the
% maximum return time entropy as well as the probability transition matrix
% correlated to the given graph adjcent matrix, weighted matrix and
% stationary distribution. epsilon denotes the lower bound of non-zero
% parameter in probability transition matrix, and eta is the truncated
% parameter. 
% 
% In this function we use fmincon to do optimization.
% 
% The mathematical form can be found in 
% <https://ieeexplore.ieee.org/abstract/document/8675541>
% 
% Example
%   A=[1 1 0;1 0 1;0 1 1];
%   PI=[1/6;1/2;1/3];
%   W=[1 2 0;3 0 4;0 5 6];
%   epsilon=0.08;
%   eta=0.1;
%   [P_op,f]=ReturnTimeEntropyOp(A,PI,W,epsilon,eta);
n=size(A,2);
A_inv=ones(n,n);
A_inv(A>0)=0;
A_vec=zeros(n^2,1);
A_inv_vec=zeros(n^2,1);
A_inv_vec1=zeros(n^2,1);
A_eq1=zeros(n,n^2);
for i=1:n^2
    A_vec(i)=A(i);
    if A_inv(i)==1
        A_inv_vec(i)=A_inv(i);
        A_eq1=[A_eq1;A_inv_vec'];
        A_inv_vec=zeros(n^2,1);
        A_inv_vec1(i)=1;
        continue
    end       
end


A1=[diag(A_vec);-diag(ones(n^2,1))];
b1=[A_vec;-epsilon*A_vec-1e-6*A_inv_vec1];
for i=1:n
    for j=i:n:n^2
        A_eq1(i,j)=1;
    end
end
b_eq1=[ones(n,1);zeros(size(A_eq1,1)-n,1)];
v_place=[];
for i = 1:n
    for j = 1:n
       v_place=[v_place j+(i-1)*n];
    end
end
function J=myfun_return(PI,x,W,eta)
v_place=[];
n=size(W,2);
P=zeros(n,n);
for i = 1:n
    for j = 1:n
       v_place=[v_place j+(i-1)*n];
    end
end
P(v_place)=x;
w_max=max(max(W));
PI_min=min(PI);
N_eta=ceil(w_max/(eta*PI_min))-1;
J=0;
F=HittingTime(P,W,N_eta);
for k=1:N_eta
    for i=1:n
        if F((N_eta+w_max-k)*n+i,i)==0
            Entropy=0;
        else
            Entropy=F((N_eta+w_max-k)*n+i,i)*log(F((N_eta+w_max-k)*n+i,i));
        end
        J=J-PI(i)*Entropy;
    end
end
J=-J;
end
X0=A./repmat(sum(A,2),[1,n]);
P0=zeros(n^2,1);
P0=X0(:);
A_eq2=zeros(n,n);
b_eq2=PI;
for i=1:n
    for j=(i-1)*n+1:(i-1)*n+n
        A_eq2(i,j)=PI(j-(i-1)*n);
    end
end
A_eq2=[A_eq1;A_eq2];
b_eq2=[b_eq1;b_eq2];
fa=rand(size(P0));
fa=zeros(size(P0,1),1);
xnew = linprog(fa,A1,b1,A_eq2,b_eq2);
options = optimoptions('fmincon','Algorithm','sqp','MaxFunctionEvaluations',10000000,'MaxIterations',10000000);
[y,f] = fmincon(@(x)myfun_return(PI,x,W,eta),xnew,A1,b1,A_eq2,b_eq2,[],[],[],options);

P_op=zeros(n,n);
P_op(v_place)=y;
f=-f;
end