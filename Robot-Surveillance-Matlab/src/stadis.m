function PI=stadis(P)
%stadis(P) is a function used to calculate the stationary distribution of a
%probability transition matrix. The mathematical form can be found in 
%<https://en.wikipedia.org/wiki/Markov_chain#Steady-state_analysis_and_limiting_distributions>
%
%Example
%   P=[1/3 1/2 1/6;1/5 2/5 2/5;1/7 2/7 4/7];
%   PI=stadis(P);
[V,D]=eig(P');
tmp=1;
for i=1:size(P,2)
    if abs(single(D(i,i)-1))<=1e-6;
        tmp=i;
        break
    end
end

PI=V(:,tmp)./sum(V(:,tmp));

    