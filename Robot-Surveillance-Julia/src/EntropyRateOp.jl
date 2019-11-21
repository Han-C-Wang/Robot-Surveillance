module EntropyRateOpMod
include("stadis.jl")
# include("DataMethod.jl")
# include("DataStructure.jl")
using Base,SparseArrays,LinearAlgebra,Ipopt,JuMP
using .stadisMod
# using .DataStructureMod,.DataMethodMod
export EntropyRateOp
SuperVariable=Union{Real,VariableRef}
if !hasmethod(zero,Tuple{VariableRef})
    Base.zero(::Type{VariableRef})=0
end
try
    zero(SuperVariable)
catch
    Base.zero(::Type{SuperVariable})=0
end

"""
EntropyRateOp(A,PI) is a function to calculate
the probability ransition matrix with maximum
entropy rate corraletd to give graph and stationary
distribution.

The mathematical form can be found in
<https://ieeexplore.ieee.org/abstract/document/8371596>
#Example
```test
julia> A=[1 1 0;1 1 1;0 1 1]
3×3 Array{Int64,2}:
 1  1  0
 1  1  1
 0  1  1

julia> PI=[1/6;1/2;1/3]
3-element Array{Float64,1}:
 0.16666666666666666
 0.5
 0.3333333333333333

julia> f=EntropyRateOp(A,PI)

julia> f[1]
3×3 Array{Float64,2}:
  0.433349     0.566651  7.05297e-38
  0.188884     0.456787  0.354329
 -2.46854e-37  0.531493  0.468507

julia> f[2]
0.0
```

"""
function EntropyRateOp(A,PI)
#################recursive solution######################
# n=size(P,2)
# A=zeros(n,n)
# for i=1:n
#     for j=1:n
#         if P[i,j]>0
#             A[i,j]=1
#         end
#     end
# end
# PI=zeros(n,1)
# PI=stadis(P)
# PI_sqrt=Vector{Float64}(undef, n)
# for i=1:n
#     PI_sqrt[i]=sqrt(PI[i])
# end
# yeta=maximum(sum(A.*repeat(PI_sqrt,1,n),dims=2))
# zeta=maximum(sum(A.*repeat(PI,1,n),dims=2))
# x=PI./zeta
# onevec=Vector{Float64}(undef,n)
# for i=1:n
#     onevec[i]=1
# end
# bool=(diagm(0=>x)*A*x-PI<(1e-6)*onevec)&&(diagm(0=>x)*A*x-PI>-(1e-6)*onevec)
# # count=1
# while bool
#     x=x-1/(2*yeta)*(diagm(0=>x)*A*x-PI)
# end
#
# P_max=inv(diagm(0=>A*x))*A*diagm(0=>x)
# logPI=Vector{Float64}(undef,n)
# for i=1:n
#     if PI[i]!=0
#         logPI[i]=log(PI[i])
#     else
#         logPI[i]=0
#     end
# end
# logx=Vector{Float64}(undef,n)
# for i=1:n
#     if x[i]!=0
#         logx[i]=log(x[i])
#     else
#         logx[i]=0
#     end
# end
# xtrans=Array{Float64,2}(undef, 1,n)
# transpose!(xtrans, x)
# PItrans=Array{Float64}(undef, 1,n)
# transpose!(PItrans, PI)
# H=-2*xtrans*A*diagm(0=>x)*logx+PItrans*logPI
# return P_max,H[1,1]
##################################################################
##########################JuMP solution############################
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
    b1=vcat(ones(n^2,1),-zeros(n^2,1));
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
    function myfun(x...)
        P=Array{SuperVariable,2}(undef,n,n)
        for i=1:n*n
            if v_place[i+1]<=n*n
                if rem(v_place[i+1],n)!=0
                    P[convert(Int64,rem(v_place[i+1],n)),convert(Int64,floor(v_place[i+1]/n))+1]=x[i]
                else
                    P[n,convert(Int64,floor(v_place[i+1]/n))]=x[i]
                end
            end
        end
        f=0.0;
        for i=1:n
            for j=1:n
                if abs(P[i,j])>1e-5
                    P[i,j]=1
                    f=f+PI[i]*P[i,j]*log(P[i,j])
                end
            end
        end
        return f
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
    JuMP.optimize!(model)#开始优化
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
