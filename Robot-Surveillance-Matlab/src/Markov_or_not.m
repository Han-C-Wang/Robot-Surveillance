function Markov_or_not(P)
n=size(P,2);
for i=1:n
    if abs(single(sum(P(i,:),2))-1)>=1e-4
       error('the matrix you input is not an illegal probability transition matrix') ;
    end
end
end