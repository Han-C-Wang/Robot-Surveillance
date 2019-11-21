function F = HittingTime(P,W,tau)
% HittingTime(P,W,tau) is a function used to calculate the hitting time
% probability with predefined probability transition matrix, weighted matrix
% and duration. The mathematical form can be found in
% <https://ieeexplore.ieee.org/abstract/document/7094271>
% Example
%   P=[1/3 1/2 1/6;1/5 2/5 2/5;1/7 2/7 4/7];
%   W=[1 2 3;4 5 6;7 8 9];
%   tau=10;
%   F=HittingTime(P,W,tau);
n=size(P,2);
w_max=max(max(W));
S = zeros(n^2,n*w_max);
D = zeros(1,n^2);
K = zeros(n,n^2);

for i = 1:n
    for j = 1:n
        if W(i,j)~=0
            S(j+(i-1)*n,j+(w_max-W(i,j))*n) = 1;
        end
    end
    D(1+(i-1)*n:i*n) = (i-1)*n+1:n^2+1:n^3;
    K(i,1+(i-1)*n:i*n)=ones(1,n);
end
S = sparse(S);
K = sparse(K);
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
% end
end
