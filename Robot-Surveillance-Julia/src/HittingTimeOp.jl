module HittingTimeOpMod
# include("DataMethod.jl")
# include("DataStructure.jl")
using Base,SparseArrays,LinearAlgebra,JuMP,Ipopt
# using .DataStructureMod,.DataMethodMod
export HittingTimeOp
SuperVariable=Union{Real,VariableRef}
try
    zero(SuperVariable)
catch
    Base.zero(::Type{SuperVariable})=0
end
if !hasmethod(zero,Tuple{VariableRef})
    Base.zero(::Type{VariableRef})=0
end
"""
HittingTimeOp(A,W,tau) is a function to
calculate the optimal probability
transition matrix with maximum hitting
time probability correlated to
given graph and duration.

The mathematical form can be found in <https://ieeexplore.ieee.org/abstract/document/7094271>

#Example
```test
  A=[1 1 0;1 0 1;0 1 1]
  W=[1 2 0;3 0 4;0 5 6]
  tau=10
  f=HittingTimeOp(A,W,tau)
 ```

"""
function HittingTimeOp(A,W,tau)
n=size(A,2)
w_max=maximum((maximum(W,dims=1)),dims=2)
w_max_int=w_max[1,1]
w_max_int=convert(Int64,w_max_int)
# %%
count = 1
location_nonzero = zeros(n,n)

# v_place = []
v_place=spzeros(1,1)

# S = zeros(SuperVariable,n^2,n*w_max_int)
# D = zeros(SuperVariable,1,n^2)
# K = zeros(SuperVariable,n,n^2)
S=Array{SuperVariable,2}(undef,n^2,convert(Int64,n*w_max_int))
for i=1:n^2
    for j=1:convert(Int64,n*w_max_int)
        S[i,j]=0
    end
end
D=Array{SuperVariable,2}(undef,1,n^2)
for i=1:n^2
    D[1,i]=0
end
K=Array{SuperVariable,2}(undef,n,n^2)
for i=1:n
    for j=1:n^2
        K[i,j]=0
    end
end
for i = 1:n
    for j = 1:n
        if A[i,j] == 1
            location_nonzero[i,j] = count
            count = count + 1
            # v_place=[v_place j+(i-1)*n]
            v_place=hcat(v_place,j+(i-1)*n)
            S[j+(i-1)*n,j+(w_max_int-convert(Int64,W[i,j]))*n] = 1
        end

    end
    D[1+(i-1)*n:i*n] = (i-1)*n+1:n^2+1:n^3
    K[i,1+(i-1)*n:i*n]=ones(1,n)
end

S = sparse(S)
# K = sparse(K)
row_nonzero = sum(A,dims=2)
total_nonzero=count-1

tmp = zeros(1,total_nonzero)
tmp[1,1:convert(Int64,row_nonzero[1])] = ones(1,convert(Int64,row_nonzero[1]))
A_eq = tmp
for i = 2:n
    tmp = zeros(1,total_nonzero)
    tmp[1,convert(Int64,sum(row_nonzero[1:i-1],dims=1)[1]+1):convert(Int64,sum(row_nonzero[1:i-1],dims=1)[1]+row_nonzero[i])] = ones(1,convert(Int64,row_nonzero[i]))
    # global A_eq
    A_eq = [A_eq;tmp]
end
b_eq = ones(n,1)

A_ineq = -Matrix{Float64}(I, total_nonzero, total_nonzero)
b_ineq = zeros(total_nonzero,1)
# %% In last section, we generated a random initial point P.


# tau = 30

x0 = 1/total_nonzero*ones(total_nonzero^2,1)

#
# options = optimoptions('fmincon','Algorithm','sqp','MaxFunctionEvaluations',
# 10000000,'MaxIterations',10000000)
# [y,f] = fmincon(@(x)myfun_travel(x,W,tau,n,v_place,w_max_int,S,D,K),
# x0,A_ineq,b_ineq,A_eq,b_eq,zeros(total_nonzero,1),ones(total_nonzero,1),[],options)
function op_myfun_travel(x...)

# P=zeros(SuperVariable,n,n)
P=Array{SuperVariable,2}(undef,n,n)
for i=1:n
    for j=1:n
        P[i,j]=0
    end
end
for cnt=1:total_nonzero
    if rem(v_place[cnt+1],n)!=0
        P[convert(Int64,rem(v_place[cnt+1],n)),convert(Int64,floor(v_place[cnt+1]/n))+1]=x[cnt]
    else
        P[n,convert(Int64,floor(v_place[cnt+1]/n))]=x[cnt]
    end
end
F=zeros(SuperVariable,n*(tau+w_max_int),n)
F_cumul=zeros(SuperVariable,n,n)
F_cumul=sparse(F_cumul)

P_sparse = zeros(SuperVariable,n,n*n)
tmp=repeat(P,1,n);
for i=1:n
    for j=1:n^2
        P_sparse[i,j]=K[i,j]*tmp[i,j];
    end
end
P_sparse=sparse(P_sparse)
F=sparse(F)
W_tmp=zeros(SuperVariable,n,n)
for k = w_max_int+1:tau+w_max_int
    R = S * F[(k-1-w_max_int)*n+1:(k-1)*n,:]
    for i=1:n*n
        if D[i]<=(n*n)*n
            if rem(D[i],n*n)==0
                R[n*n,convert(Int64,floor(D[i]/(n*n)))]=0.0
            else
                R[convert(Int64,rem(D[i],(n*n))),convert(Int64,floor(D[i]/(n*n)))+1]=0.0
            end
        end
    end
    F[(k-1)*n+1:k*n,:] = P_sparse * R
    if k<=2*w_max_int
        for i=1:n
            for j=1:n
                W_tmp[i,j]=convert(Float64,W[i,j])
                if W_tmp[i,j]!=k-w_max_int
                    W_tmp[i,j]=0.0
                else
                    W_tmp[i,j]=P[i,j]
                end
            end
        end
        W_tmp=sparse(W_tmp)

        F[(k-1)*n+1:k*n,:] = F[(k-1)*n+1:k*n,:] + W_tmp
    end
    F_cumul[1:n,1:n] += F[(k-1)*n+1:k*n,:]
end
f = -minimum((minimum(F_cumul,dims=1)),dims=2)
return f[1,1]
end

function fmincon(x0,A_ineq,b_ineq,A_eq,b_eq)
    verbose=true
    model=Model(with_optimizer(Ipopt.Optimizer))

    # @variable(model,zeros(total_nonzero,1)<=x<=ones(total_nonzero,1),start= 1/n*ones(n^2,1))
    # global x,P,P_sparse
    @variable(model,x[i=1:total_nonzero],start=x0[i])
    # @variable(model,x[i=1:total_nonzero])
    register(model, :op_myfun_travel, total_nonzero, op_myfun_travel; autodiff = true)
    @NLobjective(model,Min,op_myfun_travel(x...))
    @constraint(model,A_ineq*x.<=b_ineq)
    @constraint(model,A_eq*x.==b_eq)

    JuMP.optimize!(model)#开始优化
    obj_value=JuMP.objective_value(model)
    x_value=JuMP.value.(x)
    x=zeros(n,n)
    for cnt=1:total_nonzero
        if rem(v_place[cnt+1],n)!=0
            x[convert(Int64,rem(v_place[cnt+1],n)),convert(Int64,floor(v_place[cnt+1]/n))+1]=x_value[cnt]
        else
            x[n,convert(Int64,floor(v_place[cnt+1]/n))]=x_value[cnt]
        end
    end
    if verbose
        println("Objective value: ", obj_value)
        println("x = ", x_value)
    end
    return [x',-obj_value]
end

# op_myfun_travel(x0)
fmincon(x0,A_ineq,b_ineq,A_eq,b_eq)
end
# P_out=HittingTimeOp([1 1 1;1 1 1 ;1 1 1],[1 2 3;4 5 6;7 8 9],2)
end
