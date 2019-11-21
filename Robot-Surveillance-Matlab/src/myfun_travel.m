function f = myfun_travel(x,W,tau,n,v_place,w_max,S,D,K)

P = zeros(n,n);
P(v_place) = x;

F = zeros(n*(tau+w_max),n);
F_cumul = zeros(n,n);

P_sparse = K.*repmat(P,1,n);


for k = w_max+1:2*w_max

    R = S * F((k-1-w_max)*n+1:(k-1)*n,:);

    R(D) = 0;
    F((k-1)*n+1:k*n,:) = P_sparse * R;
    F((k-1)*n+1:k*n,:) = F((k-1)*n+1:k*n,:) + P .* (W==(k-w_max));
    F_cumul = F_cumul + F((k-1)*n+1:k*n,:);

end
for k = 2*w_max+1:tau+w_max
    R = S * F((k-1-w_max)*n+1:(k-1)*n,:);
    R(D)=0;
    F((k-1)*n+1:k*n,:) = P_sparse * R;
    F_cumul = F_cumul + F((k-1)*n+1:k*n,:);
end
f = -min(min(F_cumul));



