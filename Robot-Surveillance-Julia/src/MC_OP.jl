module MC_OP_MOD
include("ReturnTimeEntropyOp.jl")
include("EntropyRateOp.jl")
include("HittingTimeOp.jl")
include("KemenyOp.jl")
include("Markov_or_not.jl")
include("Irreducible_or_not.jl")
include("MixingTimeOp.jl")
using Base,LinearAlgebra,SparseArrays,JuMP,Ipopt,Convex,SCS
using .ReturnTimeEntropyOpMod,.HittingTimeOpMod,.KemenyOpMod,.MixingTimeOpMod,.EntropyRateOpMod
export MC_OP
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
MC_OP Calculation function
MC_OP(X,Option) is a mixed function which include ReturnTimeEntropyOp, MixingTimeOp,
KemenyOp, and HittingTimeOp. All of the mathematical forms can be found
in professor Francesco Bullo's publications:
<http://motion.me.ucsb.edu/papers/index.html>.

The input parameters of the function include two parts. Firstly, the graph
properties:
  A:  Probability transition matrix                W:  Weighted matrix
  PI:  stationary distribution                     epsilon:  lower bound
  tau:  Duration                                   eta:  truncated parameter

Secondly, the result you want to get:
  return time entropy:  ReturnTimeEntropyOp          mixing time: MixingTimeOp
  kemeny:  KemenyOp                                  hitting time probability: HittingTimeOp
input order of the function is A, PI, W, tau, epsilon, eta, Option.

#Example
```
   A=[1 1 0;1 0 1;0 1 1]
   W=[1 2 3;4 5 6;7 8 9]
   tau=10
   F=MC_OP(A,W,tau,"HittingTimeOp")
```
#Other example
```
   F=MC_OP(A,W,tau,"HittingTimeOp")
   F=MC_OP(A,PI,W,"KemenyOp")
   F=MC_OP(A,"MixingTimeOp")
   F=MC_OP(A,PI,W,epsilon,yeta,"ReturnTimeEntropyOp")
   F=MC_OP(A,PI,"EntropyRateOp")
```

"""
function MC_OP(x...)
    nargin=length(x)
    if nargin==1
        error("please input your adjacent matrix and option")
    elseif nargin==2
        if !isa(x[1],Matrix)
            error("please input the adjacent matrix firstly")
        end
        Irreducible_or_not(x[1])
        A=x[1]
        if isa(x[2],Number)
            error("please input your option finally")
        else
            if size(x[1],1)!=size(x[1],2)
                error("the adjacent matrix must be a square matrix")
            end
            sumt=0
            n=size(A,2)
            for i=1:n
                for j=1:n
                    sumt=sumt+abs(A[i,j]-A'[i,j])
                end
            end
            # if sumt>1e-6
            #     error("you have input an asymetric adjacent matrix")
            # end
            if x[2]=="MixingTimeOp"
                return MixingTimeOp(A)
            elseif x[2]=="EntropyRateOp"
                error("please input the stationary distribution")
            elseif x[2]=="KemenyOp"
                error("please input the stationary distribution")
            elseif x[2]=="HittingTimeOp"
                error("please input the duration")
            elseif x[2]=="ReTurnTimeEntropyOp"
                error("please input lower bound of probability and trancated parameter")
            else
                error("please input legal options!")
            end
        end
    elseif nargin==3
        if !isa(x[1],Matrix)
            error("please input your adjacent matrix firstly")
        end
        Irreducible_or_not(x[1])
        if isa(x[3],Number)
            error("please input your option finally")
        else
            if size(x[1],1)!=size(x[1],2)
                error("the adjacent matrix must be a square matrix")
            end
            A=x[1]
            sumt=0
            n=size(A,2)
            for i=1:n
                for j=1:n
                    sumt=sumt+abs(A[i,j]-A'[i,j])
                end
            end
            # if sumt>1e-6
            #     error("you have input an asymetric adjace nt matrix")
            # end
                if x[3]=="MixingTimeOp"
                    return MixingTimeOp(A)
                elseif x[3]=="EntropyRateOp"
                    if size(x[2],2)!=1
                        error("the stationary distribution must be a column vector")
                    end
                    if abs(sum(x[2])-1)>1e-6
                        error("please input legal stationary distribution")
                    end
                    PI=x[2]
                    return EntropyRateOp(A,PI)
                elseif x[3]=="HittingTimeOp"
                    if size(x[2],1)!=1||size(x[2],2)!=1
                        error("the duration must be a scalar")
                    end
                    if round(x[2])!=x[2]||x[2]<0
                        error("the duration must be a non-negative integer")
                    end
                    tau=x[2]
                    n=size(x[1],1)
                    W=zeros(Int64,n,n)
                    for i=1:n
                        for j=1:n
                            if A[i,j]>0
                                W[i,j]=1
                            end
                        end
                    end
                    return HittingTimeOp(A,W,tau)
                elseif x[3]=="KemenyOp"
                    if size(x[2],2)!=1
                        error("the stationary distribution must be a column vector")
                    end
                    if abs(sum(x[2])-1)>1e-6
                        error("please input legal stationary distribution")
                    end
                    PI=x[2]
                    n=size(x[1],1)
                    W=zeros(Int64,n,n)
                    for i=1:n
                        for j=1:n
                            if A[i,j]>0
                                W[i,j]=1
                            end
                        end
                    end
                    return KemenyOp(A,PI,W)
                elseif x[3]=="ReturnTimeEntropyOp"
                    error("please input lower bound of probability and trancated parameter")
                else
                    error("please input legal options!")
                end
        end
    elseif nargin==4
        if !isa(x[1],Matrix)
            error("please input your adjacent matrix")
        end
        Irreducible_or_not(x[1])
        if isa(x[4],Number)
            error("please input your option finally")
        else
            if size(x[1],1)!=size(x[1],2)
                error("the adjacent matrix must be a square matrix!")
            end
            A=x[1]
            n=size(A,2)
            sumt=0
            for i=1:n
                for j=1:n
                    sumt=sumt+abs(sum(A[i,j]-A'[i,j]))
                end
            end
            # if sumt>1e-6
            #     error("you have input an asymetric adjacent matrix")
            # end
                if x[4]=="MixingTimeOp"
                    return MixingTimeOp(A);
                elseif x[4]=="EntropyRateOp"
                    if size(x[2],2)!=1
                        error("the stationary distribution must be a column vector")
                    end
                    if single(sum(x[2]))!=1
                        error("please input legal stationary distribution")
                    end
                    PI=x[2]
                    return EntropyRateOp(A,PI)
                elseif x[4]=="HittingTimeOp"
                    if size(x[2],1)!=size(x[2],2)
                        error("the weighted matrix must be a square matrix")
                    end
                    if size(x[2],1)!=size(x[1],1)
                        error("the dimension of adjacent matrix and weighted matrix doesnt match")
                    end
                    if size(x[3],1)!=1||size(x[3],2)!=1
                        error("the duration must be a scalar")
                    end
                    if round(x[3])!=x[3]||x[3]<0
                        error("the duration must be a non-negative integer")
                    end

                    W=x[2]
                    n=size(W,2)
                    for i=1:n
                        if round(W[i,i])!=W[i,i]
                            error("the diagonal entries of weighted matrix must be nonzero")
                        end
                    end
                    tmp=zeros(n,n)
                    for i=1:n
                        for j=1:n
                            if W[i,j]>0
                                tmp[i,j]=1
                            end
                        end
                    end
                    sumt=0
                    for i=1:n
                        for j=1:n
                            sumt=sumt+abs(tmp[i,j]-A[i,j])
                        end
                    end
                    if sumt!=0
                        error("the adjacent matrix and weighted matrix mismatch")
                    end

                    tau=x[3]
                    return HittingTimeOp(A,W,tau)
                elseif x[4]=="KemenyOp"
                    if size(x[3],1)!=size(x[3],2)
                        error("the weighted matrix must be a square matrix")
                    end
                    if size(x[3],1)!=size(x[1],1)
                        error("the dimension of adjacent matrix and weighted matrix doesnt match")
                    end
                    if size(x[2],2)!=1
                        error("the stationary distribution must be a column vector")
                    end
                    if abs(sum(x[2])-1)>1e-6
                        error("please input legal stationary distribution")
                    end
                    PI=x[2]
                    if size(x[3],1)!=size(x[3],2)
                        error("the weighted matrix must be a square matrix")
                    end
                    if size(x[3],1)!=size(x[1],1)
                        error("the dimension of adjacent matrix and weighted matrix doesnt match")
                    end
                    W=x[3]
                    n=size(W,2)
                    tmp=zeros(n,n)
                    for i=1:n
                        for j=1:n
                            if W[i,j]>0
                                tmp[i,j]=1
                            end
                        end
                    end
                    return KemenyOp(A,PI,W)
                elseif x[4]=="ReturnTimeEntropyOp"
                    error("please input lower bound of probability and trancated parameter")
                else
                    error("please input legal option")
                end
        end
    elseif nargin==5
        if !isa(x[1],Matrix)
            error("please input your adjacent matrix")
        end
        Irreducible_or_not(x[1])
        if isa(x[5],Number)
            error("please input your option finally")
        else
            if size(x[1],1)!=size(x[1],2)
                error("the adjacent matrix must be a square matrix!")
            end
            A=x[1]
            sumt=0
            n=size(A,2)
            for i=1:n
                for j=1:n
                    sumt=sumt+abs(A[i,j]-A'[i,j])
                end
            end
            # if sumt!=0
            #     error("you have input an asymetric adjacent matrix")
            # end
                if x[5]=="MixingTimeOp"
                    return MixingTimeOp(A)
                elseif x[5]=="EntropyRateOp"
                    if size(x[2],2)!=1
                        error("the stationary distribution must be a column vector")
                    end
                    if single(sum(x[2]))!=1
                        error("please input legal stationary distribution")
                    end
                    PI=x[2]
                    return EntropyRateOp(A,PI)
                elseif x[5]=="HittingTimeOp"
                    if size(x[3],1)!=size(x[3],2)
                        error("the weighted matrix must be a square matrix")
                    end
                    if size(x[3],1)!=size(x[1],1)
                        error("the dimension of adjacent matrix and weighted matrix doesnt match")
                    end
                    if size(x[4],1)!=1||size(x[4],2)!=1
                        error("the duration must be a scalar")
                    end
                    if round(x[4])!=x[4]||x[4]<0
                        error("the duration must be a non-negative integer")
                    end

                    W=x[3]
                    n=size(W,2)
                    for i=1:n
                        if fix(W(i,i))~=W(i,i)
                            error("the diagonal entries of weighted matrix must be nonzero")
                        end
                    end
                    tmp=zeros(n,n)
                    sumt=0
                    for i=1:n
                        for j=1:n
                            if W[i,j]>0
                                tmp[i,j]=1
                            end
                        end
                    end
                    for i=1:n
                        for j=1:n
                            sumt=sumt+abs(tmp[i,j]-A[i,j])
                        end
                    end
                    if sumt!=0
                        error("the adjacent matrix and weighted matrix mismatch")
                    end
                    tau=x[4]
                    return HittingTimeOp(A,W,tau)
                elseif x[5]=="KemenyOp"
                    if size(x[3],1)!=size(x[3],2)
                        error("the weighted matrix must be a square matrix")
                    end
                    if size(x[3],1)!=size(x[1],1)
                        error("the dimension of adjacent matrix and weighted matrix doesnt match")
                    end
                    if size(x[2],2)!=1
                        error("the stationary distribution must be a column vector")
                    end
                    if abs(sum(x[2])-1)>1e-6
                        error("please input legal stationary distribution")
                    end
                    PI=x[2]
                    if size(x[3],1)!=size(x[3],2)
                        error("the weighted matrix must be a square matrix")
                    end
                    if size(x[3],1)!=size(x[1],1)
                        error("the dimension of adjacent matrix and weighted matrix doesnt match")
                    end
                    W=x[3]
                    tmp=zeros(n,n)
                    sumt=0
                    for i=1:n
                        for j=1:n
                            if W[i,j]>0
                                tmp[i,j]=1
                            end
                        end
                    end
                    for i=1:n
                        for j=1:n
                            sumt=sumt+abs(tmp[i,j]-A[i,j])
                        end
                    end
                    if sumt!=0
                        error("the adjacent matrix and weighted matrix mismatch")
                    end
                    return KemenyOp(A,PI,W)
                elseif x[5]=="ReturnTimeEntropyOp"
                    if size(x[2],2)!=1
                        error("the stationary distribution must be a column vector")
                    end
                    if abs(sum(x[2])-1)>1e-6
                        error("please input legal stationary distribution")
                    end
                    PI=x[2]
                    if size(x[3],1)!=1||size(x[3],2)!=1
                        error("the lower bound of probability must be a scalar")
                    end
                    if  x[3]<0||x[3]>1
                        error("the duration must be a non-negative and smaller than one")
                    end
                    if size(x[4],1)!=1||size(x[4],2)!=1
                        error("the trancated parameter must be a scalar")
                    end
                    if  x[4]<0||x[4]>1
                        error("the duration must be a non-negative and smaller than one")
                    end
                    epsilon=x[3]
                    eta=x[4]
                    n=size(A,2)
                    W=zeros(n,n)
                    for i=1:n
                        for j=1:n
                            if A[i,j]>0
                                W[i,j]=1
                            end
                        end
                    end
                    return ReturnTimeEntropyOp(A,PI,W,epsilon,eta)
                else
                    error("please input legal option")
                end
        end
    elseif nargin==6
        if !isa(x[1],Matrix)
            error("please input your adjacent matrix")
        end
        Irreducible_or_not(x[1])
        if isa(x[6],Number)
            error("please input your option finally")
        else
            if size(x[1],1)!=size(x[1],2)
                error("the adjacent matrix must be a square matrix!")
            end
            A=x[1]
            sumt=0
            n=size(A,2)
            for i=1:n
                for j=1:n
                    sumt=sumt+abs(A[i,j]-A'[i,j])
                end
            end
            # if sumt!=0
            #     error("you have input an asymetric adjacent matrix")
            # end

                if x[6]=="HittingTimeOp"
                    error("you have input too many parameters!")
                elseif x[6]=="KemenyOp"
                    error("you have input too many parameters!")
                elseif x[6]=="EntropyRateOp"
                    error("you have input too many parameters!")
                elseif x[6]=="ReturnTimeEntropyOp"
                    PI=x[2]
                    if abs(sum(x[2])-1)>1e-6
                        error("please input legal stationary distribution")
                    end
                    W=x[3]
                    if size(x[3],1)!=size(x[3],2)
                        error("the weighted matrix must be a square matrix")
                    end
                    if size(x[3],1)!=size(x[1],1)
                        error("the dimension of adjacent matrix and weighted matrix doesnt match")
                    end
                    epsilon=x[4]
                    eta=x[5]
                    return ReturnTimeEntropyOp(A,PI,W,epsilon,eta)
                else
                    error("please input legal option")
            end

        end

    else
        error("you have input too many parameters")
    end
end
end
