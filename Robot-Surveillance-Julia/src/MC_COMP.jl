module MC_COMP_MOD
include("ReturnTimeEntropyEva.jl")
include("EntropyRateEva.jl")
include("HittingTimeEva.jl")
include("KemenyEva.jl")
include("Markov_or_not.jl")
include("Irreducible_or_not.jl")
include("MixingTimeEva.jl")
include("stadis.jl")
using .ReturnTimeEntropyEvaMod,.HittingTimeEvaMod,.KemenyEvaMod,.MixingTimeEvaMod,.stadisMod,.EntropyRateEvaMod
using Base,LinearAlgebra,SparseArrays
export MC_COMP
"""
MC_COMP Calculation function
MC_COMP(X,Option) is a function which is able to compute the stationary
distribution, the Kemeny constant, the mixing time, the hitting time
distribtion, the entropy rate and the return time entropy. All of the mathematical forms can be found
in professor Francesco Bullo's publications:
<http://motion.me.ucsb.edu/papers/index.html>.

The input parameters of the function include two parts. Firstly, the graph
properties:
  P:  Probability transition matrix                W:  Weighted matrix
  tau:  Duration                                   eta:  truncated parameter

Secondly, the result you want to get:
  return time entropy:  ReturnTimeEntropy          mixing time:  MixingTime
  kemeny:  Kemeny                                  hitting time probability: HittingTime
  stationary distribution:  stadis
input order of the function is P, W, tau, yeta, Option.

#Example
```test
   P=[1/3 1/2 1/6;1/5 2/5 2/5;1/7 2/7 4/7]
   W=[1 2 3;4 5 6;7 8 9]
   tau=10
   F=MC_COMP(P,W,tau,"HittingTime")
```
Other example
```
   F=MC_COMP(P,W,tau,"HittingTime")
   F=MC_COMP(P,"EntropyRate")
   F=MC_COMP(P,W,"Kemeny")
   F=MC_COMP(P,W,yeta,"ReturnTimeEntropy")
   F=MC_COMP(P,"stadis")
```
"""
function MC_COMP(x...)
    nargin=length(x)
    if nargin==1
        error("please input your probability transition matrix and option")
    elseif nargin==2
        if !isa(x[1],Matrix)
            error("please input your probability transition matrix firstly")
        else
            P=x[1]
            if !isa(x[1],Matrix)
                error("please input your probability transition matrix firstly")
            end
            Markov_or_not(P)
            Irreducible_or_not(P)
        end
            if x[2]=="stadis"
                K=stadis(P)
            elseif x[2]=="MixingTime"
                tmp=P-P'
                sumt=0
                n=size(P,2)
                for i=1:n
                    for j=1:n
                        sumt=sumt+abs(tmp[i,j])
                    end
                end
                if sumt>1e-6
                    error("please input a symmetric probability transition matrix")
                end
                K=MixingTimeEva(P)
            elseif x[2]=="EntropyRate"
                K=EntropyRateEva(P)
            elseif x[2]=="Kemeny"
                n=size(P,2)
                W=zeros(Int64,n,n)
                for i=1:n
                    for j=1:n
                        if P[i,j]>0
                            W[i,j]=1
                        end
                    end
                end
                K=KemenyEva(P,W)
            elseif x[2]=="HittingTime"
                error("please input your duration")
            elseif x[2]=="ReturnTimeEntropy"
                error("please input your yeta")
            else
                error("please input legal option")
            end
    elseif nargin==3
        if !isa(x[1],Matrix)
            error("please input your probability transition matrix firstly")
        else
            P=x[1]
            Markov_or_not(P)
            Irreducible_or_not(P)
        end
            if x[3]=="stadis"
                K=stadis(P)
            elseif x[3]=="MixingTime"
                tmp=P-P'
                sumt=0
                n=size(P,2)
                for i=1:n
                    for j=1:n
                        sumt=sumt+abs(tmp[i,j])
                    end
                end
                if sumt>1e-6
                    error("please input a symmetric probability transition matrix")
                end
                K=MixingTimeEva(P)
            elseif x[3]=="Entropyrate"
                K=EntropyRateEva(P)
            elseif x[3]=="ReturnTimeEntropy"
                if !isa(x[2],Number)
                    error("please input the truncated parameter")
                end
                if size(x[2],1)!=1||size(x[2],2)!=1
                    error("the truncated parameter must be a scalar")
                end
                yeta=x[2]
                n=size(P,2)
                W=zeros(Int64,n,n)
                for i=1:n
                    for j=1:n
                        if P[i,j]>0
                            W[i,j]=1
                        end
                    end
                end
                K=ReturnTimeEntropyEva(P,W,yeta)
            elseif x[3]=="HittingTime"
                if !isa(x[2],Number)
                    error("please input the duration")
                end
                if size(x[2],1)!=1||size(x[2],2)!=1
                    error("the duration must be a scalar")
                end
                if round(x[2])!=x[2]||x[2]<0
                    error("the duration must be a non-negative integer")
                end
                tau=x[2]
                n=size(P,2)
                W=zeros(Int64,n,n)
                for i=1:n
                    for j=1:n
                        if P[i,j]>0
                            W[i,j]=1
                        end
                    end
                end
                F=HittingTimeEva(P,W,tau)
                n=size(W,2)
                w_max=maximum(maximum(W))
                K=zeros(n,n,tau)
                for i=1:tau
                    K[:,:,i]=F[(i+w_max-1)*n+1:(i+w_max)*n,:]
                end
            elseif x[3]=="Kemeny"
                if size(x[2],1)!=size(x[2],2)
                    error("the weighted matrix must be a square matrix")
                end
                if size(x[2],1)!=size(x[1],1)
                    error("the dimension of probability transition matrix and weighted matrix doesnt match")
                end
                W=x[2]
                K=KemenyEva(P,W)
            else
                error("please input legal option")
            end
    elseif nargin==4
        if !isa(x[1],Matrix)
            error("please input your probability transition matrix firstly")
        else
            P=x[1]
            Markov_or_not(P)
            Irreducible_or_not(P)
        end
            if x[4]=="stadis"
                K=stadis(P)
            elseif x[4]=="MixingTime"
                tmp=P-P'
                sumt=0
                n=size(P,2)
                for i=1:n
                    for j=1:n
                        sumt=sumt+abs(tmp[i,j])
                    end
                end
                if sumt>1e-6
                    error("please input a symmetric probability transition matrix")
                end
                    K=MixingTimeEva(P)
            elseif x[4]=="Entropyrate"
                K=EntropyRateEva(P)
            elseif x[4]=="ReturnTimeEntropy"
                if size(x[2],1)!=size(x[2],2)
                    error("the weighted matrix must be a square matrix")
                end
                if size(x[2],1)!=size(x[1],1)
                    error("the dimension of probability transition matrix and weighted matrix doesnt match")
                end
                if !isa(x[3],Number)
                    error("please input the truncated parameter")
                end
                if size(x[3],1)!=1||size(x[3],2)!=1
                    error("the truncated parameter must be a scalar")
                end
                yeta=x[3]
                W=x[2]
                K=ReturnTimeEntropyEva(P,W,yeta)
            elseif x[4]=="HittingTime"
                if size(x[2],1)!=size(x[2],2)
                    error("the weighted matrix must be a square matrix")
                end
                if size(x[2],1)!=size(x[1],1)
                    error("the dimension of probability transition matrix and weighted matrix doesnt match")
                end
                if !isa(x[3],Number)
                    error("please input the duration")
                end
                if size(x[3],1)!=1||size(x[3],2)!=1
                    error("the duration must be a scalar")
                end
                if round(x[3])!=x[3]||x[3]<0
                    error("the duration must be a non-negative integer")
                end
                tau=x[3]
                W=x[2]
                F=HittingTimeEva(P,W,tau)
                n=size(W,2)
                w_max=maximum(maximum(W))
                K=zeros(n,n,tau)
                for i=1:tau
                    K[:,:,i]=F[(i+w_max-1)*n+1:(i+w_max)*n,:]
                end
            elseif x[4]=="Kemeny"
                if size(x[2],1)!=size(x[2],2)
                    error("the weighted matrix must be a square matrix")
                end
                if size(x[2],1)!=size(x[1],1)
                    error("the dimension of probability transition matrix and weighted matrix doesnt match")
                end
                W=x[2]
                K=KemenyEva(P,W)
            else
                error("please input legal option")
            end
    else
        error("you have input too many parameters")
    end
    return K
end
end
