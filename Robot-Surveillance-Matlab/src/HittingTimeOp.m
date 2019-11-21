function [P_op,f]=HittingTimeOp(A,W,tau)
% HittingTimeOp(A,W,tau) is a function to calculate the optimal probability
% transition matrix with maximum hitting time probability correlated to
% given graph and duration. 
% 
% The mathematical form can be found in <https://ieeexplore.ieee.org/abstract/document/7094271>
% 
% Example
%   A=[1 1 1;1 1 1;1 1 1];
%   W=[1 2 3;3 3 4;9 5 6];
%   tau=10;
%   [P_op,f]=HittingTimeOp(A,W,tau);
n=size(A,2);
w_max=max(max(W));

count = 1;
location_nonzero = zeros(n,n);
v_place = [];

S = zeros(n^2,n*w_max);
D = zeros(1,n^2);
K = zeros(n,n^2);

for i = 1:n
    for j = 1:n
        if A(i,j) == 1
            location_nonzero(i,j) = count;
            count = count + 1;
            v_place=[v_place j+(i-1)*n];
            S(j+(i-1)*n,j+(w_max-W(i,j))*n) = 1;
         end
    end
    D(1+(i-1)*n:i*n) = (i-1)*n+1:n^2+1:n^3;
    K(i,1+(i-1)*n:i*n)=ones(1,n);
end
S = sparse(S);
K = sparse(K);

row_nonzero = sum(A,2);

total_nonzero=count-1;

tmp = zeros(1,total_nonzero);
tmp(1,1:row_nonzero(1)) = ones(1,row_nonzero(1));
A_eq = tmp;
for i = 2:n
    tmp = zeros(1,total_nonzero);
    tmp(1,sum(row_nonzero(1:i-1))+1:sum(row_nonzero(1:i-1))+row_nonzero(i)) = ones(1,row_nonzero(i));
    A_eq = [A_eq;tmp];
end
b_eq = ones(n,1);

A_ineq = -eye(total_nonzero);
b_ineq = zeros(total_nonzero,1);



fa=rand(total_nonzero,1);
xnew = linprog(fa,A_ineq,b_ineq,A_eq,b_eq);

options = optimoptions('fmincon','Algorithm','sqp','MaxFunctionEvaluations',10000000,'MaxIterations',10000000);
[y,f] = fmincon(@(x)myfun_travel(x,W,tau,n,v_place,w_max,S,D,K),xnew,A_ineq,b_ineq,A_eq,b_eq,zeros(total_nonzero,1),ones(total_nonzero,1),[],options);
f=-f;
P_op=zeros(n,n);
P_op(v_place)=y;



