function [P_op,f]=MixingTimeOp(A)
% MixingTimeOp(A) is a function to calculate the probability transition 
% matrix with maximum SLEM correlated to the given grah adjcent matrix.
% 
% In this function we use CVX to solve a sqp form problem. Usage instruction
% of CVX can be found in <http://cvxr.com/cvx/>
% 
% The mathematical form can be found in <https://web.stanford.edu/~boyd/papers/pdf/fmmc.pdf>
% 
% Example
%   A=[1 1 1 0;1 0 1 1;1 1 1 1;0 1 1 0];
%   P_op=MixingTimeOp(A);

n=size(A,2);
A_inv=ones(n,n);
A_inv(A>0)=0;

cvx_begin sdp
variable P(n,n) symmetric
variable s
minimize s;
subject to
for i=1:n
    P(:,i)>=0;
end
P-(1/n)*ones(n,1)*ones(1,n)+s*eye(n)>=0;
-P+(1/n)*ones(n,1)*ones(1,n)+s*eye(n)>=0;
P*ones(n,1) == ones(n,1);
P.*A_inv == 0;
cvx_end
P_op=P;
f=MixingTime(P);
end