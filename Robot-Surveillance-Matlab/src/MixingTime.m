function f=MixingTime(P)
% MixingTime(P) is a function used to calculate the SLEM of a probability
% transition matrix. The mathematical form can be found in
% <https://web.stanford.edu/~boyd/papers/pdf/fmmc.pdf>
% 
% The input parameter of the function is a symmetric probability transition matrix.
% 
% Example
%   P=[1/3 1/2 1/6;1/5 2/5 2/5;1/7 2/7 4/7];
%   f=MixingTime(P);
n=size(P,2);
P_one=P-1/n*ones(n,n);
f=norm(P_one,2);
end