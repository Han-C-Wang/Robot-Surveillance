function f=EntropyRate(P)
% EntropyRate(P) is a function used to calculate the entropy rate of a
% probability transition matrix. The mathematical form can be found in 
% <https://ieeexplore.ieee.org/abstract/document/8371596>
% 
% Example
%   P=[1/3 1/2 1/6;1/5 2/5 2/5;1/7 2/7 4/7];
%   f=EntropyRate(P);
n=size(P,2);
PI=stadis(P);
PI=repmat(PI,[1,n]);
P(P==0)=1;
f=-sum(sum(PI.*P.*log(P)));
f=real(f);
end