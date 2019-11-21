module HittingTimeEvaMod
# include("DataMethod.jl")
# include("DataStructure.jl")
using Base,SparseArrays,LinearAlgebra,JuMP
# using .DataStructureMod,.DataMethodMod
export HittingTimeEva
SuperVariable=Union{VariableRef,Real}
if !hasmethod(zero,Tuple{VariableRef})
    Base.zero(::Type{VariableRef})=0
end
try
    zero(SuperVariable)
catch
    Base.zero(::Type{SuperVariable})=0
end
"""
HittingTime(P,W,tau) is a function used
to calculate the hitting time
probability with predefined probability
transition matrix, weighted matrix
and duration. The mathematical form can be found in
<https://ieeexplore.ieee.org/abstract/document/7094271>
#Example
```test
julia> P=[1/3 1/2 1/6;1/5 2/5 2/5;1/7 2/7 4/7]
3×3 Array{Float64,2}:
 0.333333  0.5       0.166667
 0.2       0.4       0.4
 0.142857  0.285714  0.571429

julia> W=[1 2 3;4 5 6;7 8 9]
3×3 Array{Int64,2}:
 1  2  3
 4  5  6
 7  8  9

julia> tau=10
10

julia> f=HittingTimeEva(P,W,tau)
57×3 SparseArrays.SparseMatrixCSC{Union{Real, VariableRef},Int64} with 39 stored entries:
  [28,  1]  =  0.333333
  [38,  1]  =  0.2
  [43,  1]  =  0.1
  [48,  1]  =  0.142857
  [53,  1]  =  0.08
  [55,  1]  =  0.0238095
  ⋮
  [52,  3]  =  0.083562
  [53,  3]  =  0.0037037
  [54,  3]  =  0.571429
  [55,  3]  =  0.0334095
  [56,  3]  =  0.00123457
  [57,  3]  =  0.0238095
 ```

"""
function HittingTimeEva(P,W,tau)
    n=size(P,2)
    w_max=maximum(maximum(W))
    w_max_int=w_max[1,1]
    w_max_int=convert(Int64,w_max_int)
    # S = zeros(SuperVariable,n^2,n*w_max_int)
    S=Array{SuperVariable,2}(undef,n^2,convert(Int64,n*w_max_int))
    for i=1:n^2
        for j=1:convert(Int64,n*w_max_int)
            S[i,j]=0
        end
    end
    # D = zeros(SuperVariable,1,n^2)
    D=Array{SuperVariable,2}(undef,1,n^2)
    for i=1:n^2
        D[1,i]=0
    end
    # K = zeros(SuperVariable,n,n^2)
    K=Array{SuperVariable,2}(undef,n,n^2)
    for i=1:n
        for j=1:n^2
            K[i,j]=0
        end
    end

    for i = 1:n
        for j = 1:n
            if W[i,j]!=0
                S[j+(i-1)*n,j+convert(Int64,(w_max_int-W[i,j]))*n] = 1
            end
        end
        D[1+(i-1)*n:i*n] = (i-1)*n+1:n^2+1:n^3
        K[i,1+(i-1)*n:i*n]=ones(1,n)
    end
S = sparse(S)
K = sparse(K)


W_tmp=zeros(n,n)
# F2 = zeros(SuperVariable,n*(tau+w_max_int),n)
F2=Array{SuperVariable,2}(undef,n*(tau+w_max_int),n)
for i=1:n*(tau+w_max_int)
    for j=1:n
        F2[i,j]=0
    end
end
# F_cumul = zeros(SuperVariable,n,n)
F_cumul=Array{SuperVariable,2}(undef,n,n)
for i=1:n
    for j=1:n
        F_cumul[i,j]=0
    end
end
P_sparse = K.*repeat(P,1,n)
F2=sparse(F2)
for k = w_max_int+1:2*w_max_int
    R = S*F2[(k-1-w_max_int)*n+1:(k-1)*n,:]

    for i=1:n*n
        if D[i]<=(n*n)*n&&D[i]!=0
            if rem(D[i],n*n)==0
                R[n*n,convert(Int64,floor(D[i]/(n*n)))]=0
            end
            if rem(D[i],n*n)!=0
                R[convert(Int64,rem(D[i],(n*n))),convert(Int64,floor(D[i]/(n*n)))+1]=0
            end
        end
    end
        # break
    F2[(k-1)*n+1:k*n,:] = P_sparse*R
    for i=1:n
        for j=1:n
            W_tmp[i,j]=W[i,j]
            if W_tmp[i,j]!=k-w_max_int
                W_tmp[i,j]=0
            end
            if W_tmp[i,j]==k-w_max_int
                W_tmp[i,j]=1
            end
        end
    end
    F2[(k-1)*n+1:k*n,:] = F2[(k-1)*n+1:k*n,:] + P .* W_tmp
# %     F((k-1)*n+1:k*n,:) = F((k-1)*n+1:k*n,:) + P .* Q{k-w_max}
    F_cumul = F_cumul + F2[(k-1)*n+1:k*n,:]
end

for k = 2*w_max_int+1:tau+w_max_int
    R = S*F2[(k-1-w_max_int)*n+1:(k-1)*n,:]

    for i=1:n*n
        if D[i]<=(n*n)*n&&D[i]!=0
            if rem(D[i],n*n)==0
                R[n*n,convert(Int64,floor(D[i]/(n*n)))]=0
            end
            if rem(D[i],n*n)!=0
                R[convert(Int64,rem(D[i],(n*n))),convert(Int64,floor(D[i]/(n*n)))+1]=0
            end
        end
    end
    # break
    F2[(k-1)*n+1:k*n,:] = P_sparse * R
    F_cumul = F_cumul + F2[(k-1)*n+1:k*n,:]
end
return F2
end
end

# %% baseline
#
# % F = zeros(n,n,tau);
# % F(:,:,1) = P;
# %
# % for k = 2:tau
# %     F(:,:,k) = P * (F(:,:,k-1) - diag(diag(F(:,:,k-1))));
# % end
# % F_cumul = sum(F,3);
#
# %% baseline with sum
#
# % F = zeros(n,n,tau);
# % F(:,:,1) = P;
# % F_cumul = P;
# %
# % for k = 2:tau
# %     F(:,:,k) = P * (F(:,:,k-1) - diag(diag(F(:,:,k-1))));
# %     F_cumul = F_cumul + F(:,:,k);
# % end
# % F_cumul
#
# %% new data structure 2 half row by row operation
# % F = zeros(n*(tau+w_max),n);
# % F_cumul = zeros(n,n);
# % for k = w_max+1:2*w_max
# %     R = S * F((k-1-w_max)*n+1:(k-1)*n,:);
# %     for i = 1:n
# %         F((k-1)*n+i,:) = P(i,:) * (R(1+(i-1)*n:i*n,:) - diag(diag(R(1+(i-1)*n:i*n,:))));
# %         F((k-1)*n+i,:) = F((k-1)*n+i,:) + P(i,:) .* (W(i,:)==(k-w_max));
# %     end
# %     F_cumul = F_cumul + F((k-1)*n+1:k*n,:);
# % end
# %
# % for k = 2*w_max+1:tau+w_max
# %     R = S * F((k-1-w_max)*n+1:(k-1)*n,:);
# %     for i = 1:n
# %         F((k-1)*n+i,:) = P(i,:) * (R(1+(i-1)*n:i*n,:) - diag(diag(R(1+(i-1)*n:i*n,:))));
# %     end
# %     F_cumul = F_cumul + F((k-1)*n+1:k*n,:);
# % end
#
#
# % F_cumul
# %% method 1
# % F = zeros(n,n,tau);
# % F(:,:,1) = P .* (W == ones(n,n));
# %
# % for k = 2:tau
# %     for i = 1:n
# %         for j = 1:n
# %             for s = 1:n
# %                 if s ~= j && W(i,s) < k
# %                     F(i,j,k) = F(i,j,k) + P(i,s) * F(s,j,k-W(i,s));
# %                 end
# %                 if s == j && W(i,s) == k
# %                     F(i,j,k) = F(i,j,k) + P(i,j);
# %                 end
# %             end
# %         end
# %     end
# % end
# % sum(F,3)
#
# %% method_new dot product
# % global w_max
# % global S
# % F = zeros(n,n,tau+w_max);
# %
# % for k = w_max+1:2*w_max
# %     for i = 1:n
# %         R = S(:,:,1:n) .* F(:,:,k-W(i,:));
# %         F(i,:,k) = P(i,:) * sum(R,3);
# %         F(i,:,k) = F(i,:,k) + P(i,:) .* (W(i,:)==(k-w_max));
# %     end
# % end
# %
# % for k = 2*w_max+1:tau
# %     for i = 1:n
# %         R = S(:,:,1:n) .* F(:,:,k-W(i,:));
# %         F(i,:,k) = P(i,:) * sum(R,3);
# %     end
# % end
#
# %% method_new2 matrix product
# % F = zeros(n,n,tau+w_max);
# % for k = w_max+1:2*w_max
# %     for i = 1:n
# %         R = S * reshape(permute(F(:,:,k-W(i,:)),[3 2 1]),n^2,n);
# %         F(i,:,k) = P(i,:) * (R - diag(diag(R)));
# %         F(i,:,k) = F(i,:,k) + P(i,:) .* (W(i,:)==(k-w_max));
# %     end
# % end
# %
# % for k = 2*w_max+1:tau
# %     for i = 1:n
# %         R = S* reshape(permute(F(:,:,k-W(i,:)),[3 2 1]),n^2,n);
# %         F(i,:,k) = P(i,:) * (R - diag(diag(R)));
# %     end
# % end
#
#
# % F = zeros(n,n,tau+w_max);
# % for k = w_max+1:tau
# %     if k <= 2*w_max
# %         for i = 1:n
# %             R = S * reshape(permute(F(:,:,k-W(i,:)),[3 2 1]),n^2,n);
# %             F(i,:,k) = P(i,:) * (R - diag(diag(R)));
# %             F(i,:,k) = F(i,:,k) + P(i,:) .* (W(i,:)==(k-w_max));
# %         end
# %     else
# %         for i = 1:n
# %             R = S* reshape(permute(F(:,:,k-W(i,:)),[3 2 1]),n^2,n);
# %             F(i,:,k) = P(i,:) * (R - diag(diag(R)));
# %         end
# %     end
# % end
#
#
# %% method 3
# % F = zeros(n,n,tau);
# % F(:,:,1) = P .* (W == ones(n,n));
# % w_max = max(max(W));
#
# % W_class
# % for k = 2:w_max
# %     for i = 1:n
# %         for j = 1:n
# %             for s = 1:n
# %                 if s ~= j && W(i,s) < k
# %                     F(i,j,k) = F(i,j,k) + P(i,s) * F(s,j,k-W(i,s));
# %                 end
# %                 if s == j && W(i,s) == k
# %                     F(i,j,k) = F(i,j,k) + P(i,j);
# %                 end
# %             end
# %         end
# %     end
# % end
#
#
# % R = zeros(n,n);
# % for k = w_max+1:tau
# %     for i = 1:n
# %         for j = 1:n
# %             R(j,:) = F(j,:,k-W(i,j));
# %         end
# %         F(i,:,k) = P(i,:) * (R-diag(diag(R)));
# %     end
# % end
#
#
#
# % W_class = zeros(n,n,w_max);
# % for i = 1:n
# %     W_class(:,i,i) = ones(n,1);
# %     W_class(i,i,i) = 0;
# % end
# % for k = w_max+1:tau
# %
# %     for i = 1:n
# %         RR = F(:,:,k-W(i,:)) .* W_class(:,:,1:n); %already taken care of diagonals
# %
# %         R= sum(RR,3);
# %         F(i,:,k) = P(i,:) * R;
# %     end
# %
# % end
#
#
#
# %% method 4
# % F = zeros(n,n,tau);
# % F(:,:,1) = P .* (W == ones(n,n));
#
# % W_class
# % for k = 2:w_max
# %     for i = 1:n
# %         for j = 1:n
# %             for s = 1:n
# %                 if s ~= j && W(i,s) < k
# %                     F(i,j,k) = F(i,j,k) + P(i,s) * F(s,j,k-W(i,s));
# %                 end
# %                 if s == j && W(i,s) == k
# %                     F(i,j,k) = F(i,j,k) + P(i,j);
# %                 end
# %             end
# %         end
# %     end
# % end
#
#
# % R = zeros(n,n);
#
# % for k = 2:tau
# %     for i = 1:n
# %         R= zeros(n,n);
# %         for j = 1:n
# %             if k > W(i,j)
# %                 R(j,:) = F(j,:,k-W(i,j));
# %             end
# %         end
# %         F(i,:,k) = P(i,:) * (R-diag(diag(R)));
# %     end
# % end
#
#
# % for k = w_max+1:tau
# %     for i = 1:n
# %         for j = 1:n
# %             R(j,:) = F(j,:,k-W(i,j));
# %         end
# %         F(i,:,k) = P(i,:) * (R-diag(diag(R)));
# %     end
# % end
