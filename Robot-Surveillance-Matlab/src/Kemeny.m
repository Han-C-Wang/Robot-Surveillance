function K=Kemeny(P,W)
% Kemeny(P,W) is a function used to calculate the kemeny constant with
% predefined probability transition matrix and weighted matrix. The
% mathematical form can be found in <https://ieeexplore.ieee.org/abstract/document/7094271>
% 
% Example
%   P=[1/3 1/2 1/6;1/5 2/5 2/5;1/7 2/7 4/7];
%   W=[1 2 3;4 5 6;7 8 9];
%   K=Kemeny(P,W);
PI=stadis(P);
eignvalue=eig(P);
n=size(P,2);
K=1;
for i=1:size(P,2)
    if single(eignvalue(i))~=1
        K=K+1/(1-eignvalue(i));
    end
end
K=PI'*(P.*W)*ones(n,1)*K;
end