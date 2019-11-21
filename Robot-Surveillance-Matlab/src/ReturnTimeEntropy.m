function J=ReturnTimeEntropy(P,W,eta)
% ReturnTimeEntropy(P,W,eta) is a function to calculate the return time
% entropy with predefined probability transition matriix, weighted matrix, and truncated parameter.
% The mathematical form can be found in <https://ieeexplore.ieee.org/abstract/document/8675541>
% 
% Example
%   P=[1/3 1/2 1/6;1/5 2/5 2/5;1/7 2/7 4/7];
%   W=[1 2 3;4 5 6;7 8 9];
%   eta=0.01;
%   J=ReturnTimeEntropy(P,W,eta);
n=size(W,2);
PI=stadis(P);
w_max=max(max(W));
PI_min=min(PI);
N_eta=ceil(w_max/(eta*PI_min))-1;
J=0;
F=HittingTime(P,W,N_eta);
for k=1:N_eta
    for i=1:n
        if F(k*n+i,i)==0
            Entropy=0;
        else
            Entropy=F(k*n+i,i)*log(F(k*n+i,i));
        end
        J=J-PI(i)*Entropy;
    end
end     
end