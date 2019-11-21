module KemenyOpMod
# include("DataMethod.jl")
# include("DataStructure.jl")
export KemenyOp
using Base,SparseArrays,LinearAlgebra,Convex,SCS,JuMP,Ipopt
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
"""
KemenyOp(A,PI,W) is a function used to calculate the probability
transition matrix with maxmimum kemeny constant correlated to given graph
and stationary distribution.

In this function we use fmincon to solve optimization problem

The mathematical form can be found in
<https://ieeexplore.ieee.org/abstract/document/7094271>

#Example
```test
  A=[1 1 0;1 0 1;0 1 1]
  PI=[1/6;1/2;1/3]
  W=[1 2 0;3 0 4;0 5 6]
  f=KemenyOp(A,PI,W)
```
"""
function KemenyOp(A,PI,W)
n=size(PI,1);

A_inv=ones(n,n);
for i=1:n
    for j=1:n
        if W[i,j]>0
            A_inv[i,j]=0
        end
    end
end
v_place=spzeros(1,1)
for i = 1:n
    for j = 1:n
       v_place=hcat(v_place,j+(i-1)*n)
    end
end

diag_PI=diagm(0=>PI)
q=Vector{Float64}(undef,n)
sqdiag_PI=zeros(n,n)
inv_sqdiag_PI=zeros(n,n)
for i=1:n
    q[i]=sqrt(PI[i])
    sqdiag_PI[i,i]=sqrt(diag_PI[i])
    inv_sqdiag_PI[i,i]=diag_PI[i]^(-1/2)
end
# q=sqrt(PI);

qtrans=Array{Float64,2}(undef, 1,n)
transpose!(qtrans, q)
PItrans=Array{Float64,2}(undef, 1,n)
transpose!(PItrans, PI)

######################JuMP########################################
# %%%%%%%%evaluation function of NORMAL optimization%%%%%%%%%%%%%%%%%%%
    function Rushabh_eva(x...)
        X=zeros(SuperVariable,n,n);
        for i=1:n*n
            if v_place[i+1]<=n*n
                if rem(v_place[i+1],n)!=0
                    X[convert(Int64,rem(v_place[i+1],n)),convert(Int64,floor(v_place[i+1]/n))+1]=x[i]
                else
                    X[n,convert(Int64,floor(v_place[i+1]/n))]=x[i]

                end
            end
        end
        # P(v_place)=x;
        weighted=PItrans*(X.*W)*ones(n,1);
        eyen=Matrix{Float64}(I,n,n)
        matrix=eyen-sqdiag_PI*X*diag_PI^(-1/2)+q*qtrans
        # matrix=inv(matrix)
        summatrix=matrix[1,1]
        # global summatrix,matrix
        for i=2:n
            summatrix=summatrix+matrix[i,i]
        end
        f=weighted[1,1]*summatrix
        return f
    end

x0=1/n*ones(n^2,1);
X0=1/n*ones(2*n^2,1);
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
# %%%%%%%%%%%%%%%%%%%%%nonlinear constraint for NORMAL case%%%%%%%%%%%%%%%%%%%%%%
    function nonlcon1(x...)
        P_op=zeros(SuperVariable,n,n)
        for i=1:n*n
            if v_place[i+1]<=n*n
                if rem(v_place[i+1],n)!=0
                    P_op[convert(Int64,rem(v_place[i+1],n)),convert(Int64,floor(v_place[i+1]/n))+1]=x[i]
                else
                    P_op[n,convert(Int64,floor(v_place[i+1]/n))]=x[i]

                end
            end
        end
        Ptrans=Array{Real,2}(undef, n,n)
        transpose!(Ptrans, P_op)
        ceq=P_op.*repeat(PI,1,n)-Ptrans.*(repeat(PItrans,n,1))
        sum=0
        for i=1:n
            for j=1:n
                sum=sum+abs(ceq[i,j])
            end
        end
        return sum
    end

    verbose=true
    model=Model(with_optimizer(Ipopt.Optimizer))
    @variable(model,P[i=1:n^2])
    # @variable(model,s)
    register(model, :Rushabh_eva, n^2, Rushabh_eva; autodiff = true)
    JuMP.register(model, :nonlcon1, n^2, nonlcon1, autodiff=true)
    @objective(model,Min,Rushabh_eva(P...))
    # global P
    @constraint(model,A_eq*P.==b_eq)
    @constraint(model,A1*P.<=b1)

    @NLexpression(model,my_expr,nonlcon1(P...))
    @NLconstraint(model,my_constr,nonlcon1(P...)==0)
    if verbose
        print(model)
    end
    JuMP.optimize!(model)#开始优化
    obj_value=JuMP.objective_value(model)
    P_value=JuMP.value.(P)
    P_op=zeros(n,n)

    for i=1:n*n
        if v_place[i+1]<=n*n
            if rem(v_place[i+1],n)!=0
                P_op[convert(Int64,rem(v_place[i+1],n)),convert(Int64,floor(v_place[i+1]/n))+1]=P_value[i]
            else
                P_op[n,convert(Int64,floor(v_place[i+1]/n))]=P_value[i]

            end
        end
    end
    ################################################################
    # t=Variable()
    # X=Variable(n,n)
    # Y=Variable(n,n)
    # constraint1=(Y*ones(n,1)==t*ones(n,1))
    # constraint2=(Y>=0)
    # constraint3=(Y<=t*ones(n,n))
    # constraint4=(Y.*A_inv==0)
    # PI_matrix=repeat(PI,1,n)
    # constraint5=(Y.*PI_matrix==Y'.*PI_matrix')
    # constraint6=(t>=0)
    # constraint7=(PItrans*(Y.*W)*ones(n,1)==1)
    # constraint8=([t*(Matrix{Float64}(I,n,n)+q*q')-sqdiag_PI*Y*inv_sqdiag_PI Matrix{Float64}(I,n,n);
    # Matrix{Float64}(I,n,n) X] in :SDP )
    # problem=minimize(tr(X))
    # problem.constraints+=[constraint1,constraint2,constraint3,constraint4,constraint5,constraint6,constraint7,constraint8]
    # solve!(problem, SCSSolver(verbose=1))
    # return Y.value
        # options = optimoptions('fmincon','Algorithm','sqp','MaxFunctionEvaluations',10000000,'MaxIterations',10000000);
        # [y,H] = fmincon(@(x)Rushabh_eva(x,PI,W),x0,A1,b1,A_eq,b_eq,[],[],@nonlcon1,options);
        # P_op=zeros(n,n);
        # P_op(v_place)=y;
        return [P_op,obj_value]
    end
end
