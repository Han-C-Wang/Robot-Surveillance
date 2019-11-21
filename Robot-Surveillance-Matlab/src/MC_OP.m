function [P,K]=MC_OP(varargin)
% MC_OP Calculation function
% MC_OP(X,Option) is a mixed function which include ReturnTimeEntropyOp, MixingTimeOp,
% KemenyOp, and HittingTimeOp. All of the mathematical forms can be found
% in professor Francesco Bullo's publications:
% <http://motion.me.ucsb.edu/papers/index.html>.
% 
% The input parameters of the function include two parts. Firstly, the graph
% properties:
%   A:  Probability transition matrix                W:  Weighted matrix
%   PI:  stationary distribution                     epsilon:  lower bound
%   tau:  Duration                                   eta:  truncated parameter  
%   
% Secondly, the result you want to get:
%   return time entropy:  ReturnTimeEntropyOp          mixing time: MixingTimeOp
%   kemeny:  KemenyOp                                  hitting time probability: HittingTimeOp
% input order of the function is A, PI, W, tau, epsilon, eta, Option.
% 
% Example
%    A=[1 1 0;1 0 1;0 1 1];
%    W=[1 2 3;4 5 6;7 8 9];
%    tau=10;
%    [F,K]=MC_OP(P,W,tau,'HittingTimeOp');
% Other example
%    [F,K]=MC_OP(A,W,tau,'HittingTimeOp');
%    [F,K]=MC_OP(A,PI,W,'KemenyOp');
%    [F,K]=MC_OP(A,'MixingTimeOp');
%    [F,K]=MC_OP(A,PI,W,epsilon,yeta,'ReturnTimeEntropyOp');
%    [F,K]=MC_OP(A,PI,'EntropyRateOp');
%See also HittingTimeOp, KemenyOp, MixingTimeOp, ReturnTimeEntropyOp,
%EntropyRateOp
A=[];PI=[];W=[];tau=1;option='';
switch nargin
    case 1
        error('please input your adjacent matrix and option');
    case 2
        if ~isa(varargin{1},'numeric')
            error('please input the adjacent matrix firstly');
        end
        Irreducible_or_not(varargin{1});
        if ~isa(varargin{2},'char')
            error('please input your option finally');
        else
            if size(varargin{1},1)~=size(varargin{1},2)
                error('the adjacent matrix must be a square matrix');
            end
            if sum(sum(abs(varargin{1}-varargin{1}')))~=0
            	error('you have input an asymetric adjacent matrix');
            end
            A=varargin{1};
            switch varargin{2}
                case 'MixingTimeOp'
                    [P,K]=MixingTimeOp(A);
                case 'EntropyRateOp'
                    error('please input the stationary distribution');
                case 'KemenyOp'
                    error('please input the stationary distribution');
                case 'HittingTimeOp'
                    error('please input the duration');
                case 'ReTurnTimeEntropyOp'
                    error('please input lower bound of probability and trancated parameter');
                otherwise
                    error('please input legal options!');
            end
        end
    case 3
        if ~isa(varargin{1},'numeric')
            error('please input your adjacent matrix firstly');
        end
        Irreducible_or_not(varargin{1});
        if ~isa(varargin{3},'char')
            error('please input your option finally');
        else
            if size(varargin{1},1)~=size(varargin{1},2)
                error('the adjacent matrix must be a square matrix');
            end
            A=varargin{1};
            switch varargin{3}
                case 'MixingTimeOp'
                    [P,K]=MixingTimeOp(A);
                case 'EntropyRateOp'
                    if size(varargin{2},2)~=1
                        error('the stationary distribution must be a column vector');
                    end
                    if single(sum(varargin{2}))~=1
                        error('please input legal stationary distribution');
                    end
                    PI=varargin{2};
                    [P,K]=EntropyRateOp(A,PI);
                case 'HittingTimeOp'
                    if size(varargin{2},1)~=1||size(varargin{2},2)~=1
                        error('the duration must be a scalar');
                    end
                    if fix(varargin{2})~=varargin{2}||varargin{2}<0
                        error('the duration must be a non-negative integer');
                    end
                    tau=varargin{2};
                    n=size(varargin{1},1);
                    W=zeros(n,n);
                    W(A>0)=1;
                    [P,K]=HittingTimeOp(A,W,tau);
                case 'KemenyOp'
                    if size(varargin{2},2)~=1
                        error('the stationary distribution must be a column vector');
                    end
                    if single(sum(varargin{2}))~=1
                        error('please input legal stationary distribution');
                    end
                    PI=varargin{2};
                    n=size(varargin{1},1);
                    W=zeros(n,n);
                    W(A>0)=1;
                    [P,K]=KemenyOp(A,PI,W);
                case 'ReturnTimeEntropyOp'
                    error('please input lower bound of probability and trancated parameter');
                otherwise
                    error('please input legal options!');
            end
        end
    case 4
        if ~isa(varargin{1},'numeric')
            error('please input your adjacent matrix');
        end
        Irreducible_or_not(varargin{1});
        if ~isa(varargin{4},'char')
            error('please input your option finally');
        else
            if size(varargin{1},1)~=size(varargin{1},2)
                error('the adjacent matrix must be a square matrix!');
            end
            A=varargin{1};
            switch varargin{4}
                case 'MixingTimeOp'
                    [P,K]=MixingTimeOp(A);
                case 'EntropyRateOp'
                    if size(varargin{2},2)~=1
                        error('the stationary distribution must be a column vector');
                    end
                    if single(sum(varargin{2}))~=1
                        error('please input legal stationary distribution');
                    end
                    PI=varargin{2};
                    [P,K]=EntropyRateOp(A,PI);
                case 'HittingTimeOp'
                    if size(varargin{2},1)~=size(varargin{2},2)
                        error('the weighted matrix must be a square matrix');
                    end
                    if size(varargin{2},1)~=size(varargin{1},1)
                        error('the dimension of adjacent matrix and weighted matrix doesnt match');
                    end
                    if size(varargin{3},1)~=1||size(varargin{3},2)~=1
                        error('the duration must be a scalar');
                    end
                    if fix(varargin{3})~=varargin{3}||varargin{3}<0
                        error('the duration must be a non-negative integer');
                    end
                        
                    W=varargin{2};
                    n=size(W,2);
                    for i=1:n
                        if fix(W(i,i))~=W(i,i)
                            error('the diagonal entries of weighted matrix must be nonzero');
                        end
                    end
                    tmp=zeros(n,n);
                    tmp(W>0)=1;
                    if sum(sum(abs(A-tmp)))~=0
                        error('the adjacent matrix and weighted matrix mismatch');
                    end
                  
                    tau=varargin{3};
                    [P,K]=HittingTimeOp(A,W,tau);
                case 'KemenyOp'
                    if size(varargin{3},1)~=size(varargin{3},2)
                        error('the weighted matrix must be a square matrix');
                    end
                    if size(varargin{3},1)~=size(varargin{1},1)
                        error('the dimension of adjacent matrix and weighted matrix doesnt match');
                    end
                    if size(varargin{2},2)~=1
                        error('the stationary distribution must be a column vector');
                    end
                    if single(sum(varargin{2}))~=1
                        error('please input legal stationary distribution');
                    end
                    PI=varargin{2};
                    if size(varargin{3},1)~=size(varargin{3},2)
                        error('the weighted matrix must be a square matrix');
                    end
                    if size(varargin{3},1)~=size(varargin{1},1)
                        error('the dimension of adjacent matrix and weighted matrix doesnt match');
                    end
                    W=varargin{3};
                    n=size(W,2);
                    tmp=zeros(n,n);
                    tmp(W>0)=1;
                    if sum(sum(abs(A-tmp)))~=0
                        error('the adjacent matrix and weighted matrix mismatch');
                    end
                    [P,K]=KemenyOp(A,PI,W);
                case 'ReturnTimeEntropyOp'
                    error('please input lower bound of probability and trancated parameter');
                otherwise
                    error('please input legal option');
            end
        end
    case 5
        if ~isa(varargin{1},'numeric')
            error('please input your adjacent matrix');
        end  
        Irreducible_or_not(varargin{1});
        if ~isa(varargin{5},'char')
            error('please input your option finally');
        else
            if size(varargin{1},1)~=size(varargin{1},2)
                error('the adjacent matrix must be a square matrix!');
            end
            A=varargin{1};
            switch varargin{5}
                case 'MixingTimeOp'
                    [P,K]=MixingTimeOp(A);
                case 'EntropyRateOp'
                    if size(varargin{2},2)~=1
                        error('the stationary distribution must be a column vector');
                    end
                    if single(sum(varargin{2}))~=1
                        error('please input legal stationary distribution');
                    end
                    PI=varargin{2};
                    [P,K]=EntropyRateOp(A,PI);
                case 'HittingTimeOp'
                    if size(varargin{3},1)~=size(varargin{3},2)
                        error('the weighted matrix must be a square matrix');
                    end
                    if size(varargin{3},1)~=size(varargin{1},1)
                        error('the dimension of adjacent matrix and weighted matrix doesnt match');
                    end
                    if size(varargin{4},1)~=1||size(varargin{4},2)~=1
                        error('the duration must be a scalar');
                    end
                    if fix(varargin{4})~=varargin{4}||varargin{4}<0
                        error('the duration must be a non-negative integer');
                    end
                    
                    W=varargin{3};
                    n=size(W,2);
                    for i=1:n
                        if fix(W(i,i))~=W(i,i)
                            error('the diagonal entries of weighted matrix must be nonzero');
                        end
                    end
                    tmp=zeros(n,n);
                    tmp(W>0)=1;
                    if sum(sum(abs(A-tmp)))~=0
                        error('the adjacent matrix and weighted matrix mismatch');
                    end
                    tau=varargin{4};
                    [P,K]=HittingTimeOp(A,W,tau);
                case 'KemenyOp'
                    if size(varargin{3},1)~=size(varargin{3},2)
                        error('the weighted matrix must be a square matrix');
                    end
                    if size(varargin{3},1)~=size(varargin{1},1)
                        error('the dimension of adjacent matrix and weighted matrix doesnt match');
                    end
                    if size(varargin{2},2)~=1
                        error('the stationary distribution must be a column vector');
                    end
                    if single(sum(varargin{2}))~=1
                        error('please input legal stationary distribution');
                    end
                    PI=varargin{2};
                    if size(varargin{3},1)~=size(varargin{3},2)
                        error('the weighted matrix must be a square matrix');
                    end
                    if size(varargin{3},1)~=size(varargin{1},1)
                        error('the dimension of adjacent matrix and weighted matrix doesnt match');
                    end
                    W=varargin{3};
                    tmp=zeros(n,n);
                    tmp(W>0)=1;
                    if sum(sum(abs(A-tmp)))~=0
                        error('the adjacent matrix and weighted matrix mismatch');
                    end
                    [P,K]=KemenyOp(A,PI,W);
                case 'ReturnTimeEntropyOp'
                    if size(varargin{2},2)~=1
                        error('the stationary distribution must be a column vector');
                    end
                    if single(sum(varargin{2}))~=1
                        error('please input legal stationary distribution');
                    end
                    PI=varargin{2};
                    if size(varargin{3},1)~=1||size(varargin{3},2)~=1
                        error('the lower bound of probability must be a scalar');
                    end
                    if  varargin{3}<0||varargin{3}>1
                        error('the duration must be a non-negative and smaller than one');
                    end
                    if size(varargin{4},1)~=1||size(varargin{4},2)~=1
                        error('the trancated parameter must be a scalar');
                    end
                    if  varargin{4}<0||varargin{4}>1
                        error('the duration must be a non-negative and smaller than one');
                    end
                    epsilon=varargin{3};
                    eta=varargin{4};
                    n=size(A,2);
                    W=zeros(n,n);
                    W(A>0)=1;
                    [P,K]=ReturnTimeEntropyOp(A,PI,W,epsilon,eta);
                otherwise
                    error('please input legal option');
            end
        end
    case 6
        if ~isa(varargin{1},'numeric')
            error('please input your adjacent matrix');
        end  
        Irreducible_or_not(varargin{1});
        if ~isa(varargin{6},'char')
            error('please input your option finally');
        else
            if size(varargin{1},1)~=size(varargin{1},2)
                error('the adjacent matrix must be a square matrix!');
            end
            A=varargin{1};
            switch varargin{6}
                case 'HittingTimeOp'
                    error('you have input too many parameters!');
                case 'KemenyOp'
                    error('you have input too many parameters!');
                case 'EntropyRateOp'
                    error('you have input too many parameters!');
                case 'ReturnTimeEntropyOp'
                    PI=varargin{2};
                    if single(sum(varargin{2}))~=1
                        error('please input legal stationary distribution');
                    end
                    W=varargin{3};
                    if size(varargin{3},1)~=size(varargin{3},2)
                        error('the weighted matrix must be a square matrix');
                    end
                    if size(varargin{3},1)~=size(varargin{1},1)
                        error('the dimension of adjacent matrix and weighted matrix doesnt match');
                    end
                    epsilon=varargin{4};
                    eta=varargin{5};
                    [P,K]=ReturnTimeEntropyOp(A,PI,W,epsilon,eta);
                otherwise
                    error('please input legal option');
            end
                    
        end
        
    otherwise
        error('you have input too many parameters');
end             
end       
