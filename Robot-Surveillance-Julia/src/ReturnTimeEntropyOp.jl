module ReturnTimeEntropyOpMod
using Base,SparseArrays,LinearAlgebra,JuMP,Ipopt
include("stadis.jl")
include("HittingTimeEva.jl")
# include("DataMethod.jl")
# include("DataStructure.jl")
using .HittingTimeEvaMod,.stadisMod
# using .DataStructureMod,.DataMethodMod
SuperVariable=Union{Real,VariableRef}
try
    zero(SuperVariable)
catch
    Base.zero(::Type{SuperVariable})=0
end
if !hasmethod(zero,Tuple{VariableRef})
    Base.zero(::Type{VariableRef})=0
end
export ReturnTimeEntropyOp
"""
ReturnTimeEntropyOp(A,PI,W,epsilon,eta) is a function to calculate the
maximum return time entropy as well as the probability transition matrix
correlated to the given graph adjcent matrix, weighted matrix and
stationary distribution. epsilon denotes the lower bound of non-zero
parameter in probability transition matrix, and eta is the truncated
parameter.

In this function we use fmincon to do optimization.

The mathematical form can be found in
<https://ieeexplore.ieee.org/abstract/document/8675541>

#Example
```
  A=[1 1 0;1 0 1;0 1 1]
  PI=[1/6;1/2;1/3]
  W=[1 2 0;3 0 4;0 5 6]
  epsilon=0.08
  eta=0.1
  f=ReturnTimeEntropyOp(A,PI,W,epsilon,eta)
```
"""
function ReturnTimeEntropyOp(A,PI,W,epsilon,eta)
n=size(A,2);
A_inv=ones(n,n);
for i=1:n
    for j=1:n
        if A[i,j]>0
            A_inv[i,j]=0
        end
    end
end
A_vec=Vector{Float64}(undef,n^2)
A_inv_vec=zeros(n^2,1)
A_eq=zeros(n,n^2)
for i=1:n^2
    A_vec[i]=A[i]
    if A_inv[i]==1
        A_inv_vec[i]=A_inv[i]
        A_inv_vec_trans=Array{Float64,2}(undef, 1,n^2)
        transpose!(A_inv_vec_trans, A_inv_vec)
        A_eq=vcat(A_eq,A_inv_vec_trans)
        A_inv_vec=zeros(n^2,1)
        continue
    end
end
A_inv_vec_trans=Array{Float64,2}(undef, 1,n^2)
transpose!(A_inv_vec_trans, A_inv_vec)
PItrans=Array{Float64,2}(undef, 1,n)
transpose!(PItrans, PI)


b_eq=vcat(ones(n,1),zeros(size(A_eq,1)-n,1));
A1=vcat(diagm(0=>A_vec),-diagm(0=>A_vec));
b1=vcat(ones(n^2,1),-epsilon*A_vec);
for i=1:n
    for j=i:n:n^2
        A_eq[i,j]=1;
    end
end
    v_place=spzeros(1,1)
    for i = 1:n
        for j = 1:n
           v_place=hcat(v_place,j+(i-1)*n)
        end
    end
    function myfun(x...)
        P_opt=Array{SuperVariable,2}(undef,n,n)
        for i=1:n*n
            if v_place[i+1]<=n*n
                if rem(v_place[i+1],n)!=0
                    P_opt[convert(Int64,rem(v_place[i+1],n)),convert(Int64,floor(v_place[i+1]/n))+1]=x[i]
                else
                    P_opt[n,convert(Int64,floor(v_place[i+1]/n))]=x[i]

                end
            end
        end
        w_max=maximum(maximum(W))
        w_max_int=w_max[1,1]
        PI_min=minimum(PI);
        N_eta=ceil(w_max/(eta*PI_min[1,1]))-1;
        J=0;
        F=HittingTimeEva(P_opt,W,convert(Int64,N_eta));
        for k=1:convert(Int64,N_eta)
            for i=1:n
                if abs(F[k*n+i,i])<=1e-5
                    Entropy=0;
                else
                    Entropy=F[k*n+i,i]*log(F[k*n+i,i]);
                end
                J=J-PI[i]*Entropy;
            end
        end
        return J[1,1]

    end
    function nonlcon1(x...)
        P_op=Array{SuperVariable,2}(undef,n,n)
        for i=1:n*n
            if v_place[i+1]<=n*n
                if rem(v_place[i+1],n)!=0
                    P_op[convert(Int64,rem(v_place[i+1],n)),convert(Int64,floor(v_place[i+1]/n))+1]=x[i]
                else
                    P_op[n,convert(Int64,floor(v_place[i+1]/n))]=x[i]
                end
            end
        end
        tmp=PItrans*P_op-PItrans
        sum=0
        for i=1:n
            sum=sum+abs(tmp[i])
        end
        return sum


    end
    X0=1/n*ones(n^2,1)
    verbose=true
        model=Model(with_optimizer(Ipopt.Optimizer))
        @variable(model,X[i=1:n^2],start=X0[i])
        register(model, :myfun, n^2, myfun; autodiff = true)
        JuMP.register(model, :nonlcon1, n^2, nonlcon1, autodiff=true)
        @NLobjective(model,Max,myfun(X...))
        @constraint(model,A_eq*X.==b_eq)
        @constraint(model,A1*X.<=b1)
        @NLexpression(model,my_expr,nonlcon1(X...))
        @NLconstraint(model,my_constr,nonlcon1(X...)==0)
        if verbose
            print(model)
        end
        JuMP.optimize!(model)#
        obj_value=JuMP.objective_value(model)
        P_value=JuMP.value.(X)
        P_out=zeros(n,n)
        for i=1:n*n
            if v_place[i+1]<=n*n
                if rem(v_place[i+1],n)!=0
                    P_out[convert(Int64,rem(v_place[i+1],n)),convert(Int64,floor(v_place[i+1]/n))+1]=P_value[i]
                else
                    P_out[n,convert(Int64,floor(v_place[i+1]/n))]=P_value[i]
                end
            end
        end
        return [P_out,obj_value]
    end
end
