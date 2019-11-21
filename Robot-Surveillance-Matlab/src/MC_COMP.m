function K=MC_COMP(varargin)
% MC_COMP Calculation function
% MC_COMP(X,Option) is a function which is able to compute the stationary
% distribution, the Kemeny constant, the mixing time, the hitting time
% distribtion, the entropy rate and the return time entropy. All of the mathematical forms can be found
% in professor Francesco Bullo's publications:
% <http://motion.me.ucsb.edu/papers/index.html>.
% 
% The input parameters of the function include two parts. Firstly, the graph
% properties:
%   P:  Probability transition matrix                W:  Weighted matrix
%   tau:  Duration                                   eta:  truncated parameter  
%   
% Secondly, the result you want to get:
%   return time entropy:  ReturnTimeEntropy          mixing time:  MixingTime
%   kemeny:  Kemeny                                  hitting time probability: HittingTime
%   stationary distribution:  stadis
% input order of the function is P, W, tau, yeta, Option.
% 
% Example
%    P=[1/3 1/2 1/6;1/5 2/5 2/5;1/7 2/7 4/7];
%    W=[1 2 3;4 5 6;7 8 9];
%    tau=10;
%    F=MC_COMP(P,W,tau,'HittingTime');
% Other example
%    F=MC_EVA(P,W,tau,'HittingTime');
%    F=MC_EVA(P,'EntropyRate');
%    F=MC_EVA(P,W,'Kemeny');
%    F=MC_EVA(P,W,yeta,'ReturnTimeEntropy');
%    F=MC_EVA(P,'stadis')
%See also HittingTime, EntropyRate, Kemeny, ReturnTimeEntropy, stadis
switch nargin
    case 1
        error('please input your probability transition matrix and option');
    case 2
        if ~isa(varargin{1},'numeric')
            error('please input your probability transition matrix firstly');
        else
            P=varargin{1};
            if ~isa(varargin{1},'numeric')
                error('please input your probability transition matrix firstly');
            end
            Markov_or_not(P);
            Irreducible_or_not(P);
        end
        switch varargin{2}
            case 'stadis'
                K=stadis(P);
            case 'MixingTime'
                if sum(sum(abs(P-P')))>1e-6
                    error('please input a symmetric probability transition matrix');
                end
                K=MixingTime(P);
            case 'EntropyRate'
                K=EntropyRate(P);
            case 'Kemeny'
                n=size(P,2);
                W=zeros(n,n);
                W(P>0)=1;
%                 W=ones(n,n);
                K=Kemeny(P,W);
            case 'HittingTime'
                error('please input your duration');
            case 'ReturnTimeEntropy'
                error('please input your yeta');
            otherwise
                error('please input legal option');
        end      
    case 3
        if ~isa(varargin{1},'numeric')
            error('please input your probability transition matrix firstly');
        else
            P=varargin{1};
            Markov_or_not(P);
            Irreducible_or_not(P);

        end
        switch varargin{3}
            case 'stadis'
                K=stadis(P);
            case 'MixingTime'
                if sum(sum(abs(P-P')))>1e-6
                    error('please input a symmetric probability transition matrix');
                end
                K=MixingTime(P);
            case 'EntropyRate'
                K=EntropyRate(P);
            case 'ReturnTimeEntropy'
                if ~isa(varargin{2},'numeric')
                    error('please input the truncated parameter');
                end
                if size(varargin{2},1)~=1||size(varargin{2},2)~=1
                    error('the truncated parameter must be a scalar');
                end
                yeta=varargin{2};
                n=size(P,2);
                W=zeros(n,n);
                W(P>0)=1;

                K=ReturnTimeEntropy(P,W,yeta);
            case 'HittingTime'
                if ~isa(varargin{2},'numeric')
                    error('please input the duration');
                end
                if size(varargin{2},1)~=1||size(varargin{2},2)~=1
                    error('the duration must be a scalar');
                end
                if fix(varargin{2})~=varargin{2}||varargin{2}<0
                    error('the duration must be a non-negative integer');
                end
                tau=varargin{2};
                n=size(P,2);
                W=zeros(n,n);
                W(P>0)=1;

                F=HittingTime(P,W,tau);
                n=size(W,2);
                w_max=max(max(W));
                for i=1:tau
                    K(:,:,i)=F((i+w_max-1)*n+1:(i+w_max)*n,:);
                end
            case 'Kemeny'
                if size(varargin{2},1)~=size(varargin{2},2)
                    error('the weighted matrix must be a square matrix');
                end
                if size(varargin{2},1)~=size(varargin{1},1)
                    error('the dimension of probability transition matrix and weighted matrix doesnt match');
                end
                W=varargin{2};
                K=Kemeny(P,W);
            otherwise
                error('please input legal option');
        end
    case 4
        if ~isa(varargin{1},'numeric')
            error('please input your probability transition matrix firstly');
        else
            P=varargin{1};
            Markov_or_not(P);
            Irreducible_or_not(P);
        end
        switch varargin{4}
            case 'stadis'
                K=stadis(P);
            case 'MixingTime'
                if sum(sum(abs(P-P')))>1e-6
                    error('please input a symmetric probability transition matrix');
                end
                K=MixingTime(P);
            case 'EntropyRate'
                K=EntropyRate(P);
            case 'ReturnTimeEntropy'
                if size(varargin{2},1)~=size(varargin{2},2)
                    error('the weighted matrix must be a square matrix');
                end
                if size(varargin{2},1)~=size(varargin{1},1)
                    error('the dimension of probability transition matrix and weighted matrix doesnt match');
                end                
                if ~isa(varargin{3},'numeric')
                    error('please input the truncated parameter');
                end
                if size(varargin{3},1)~=1||size(varargin{3},2)~=1
                    error('the truncated parameter must be a scalar');
                end
                yeta=varargin{3};
                W=varargin{2};
                K=ReturnTimeEntropy(P,W,yeta);
            case 'HittingTime'
                if size(varargin{2},1)~=size(varargin{2},2)
                    error('the weighted matrix must be a square matrix');
                end
                if size(varargin{2},1)~=size(varargin{1},1)
                    error('the dimension of probability transition matrix and weighted matrix doesnt match');
                end                   
                if ~isa(varargin{3},'numeric')
                    error('please input the duration');
                end
                if size(varargin{3},1)~=1||size(varargin{3},2)~=1
                    error('the duration must be a scalar');
                end
                if fix(varargin{3})~=varargin{3}||varargin{3}<0
                    error('the duration must be a non-negative integer');
                end
                tau=varargin{3};
                W=varargin{2};
                F=HittingTime(P,W,tau);
                n=size(W,2);
                w_max=max(max(W));
                for i=1:tau
                    K(:,:,i)=F((i+w_max-1)*n+1:(i+w_max)*n,:);
                end
            case 'Kemeny'
                if size(varargin{2},1)~=size(varargin{2},2)
                    error('the weighted matrix must be a square matrix');
                end
                if size(varargin{2},1)~=size(varargin{1},1)
                    error('the dimension of probability transition matrix and weighted matrix doesnt match');
                end 
                W=varargin{2};
                K=Kemeny(P,W);
            otherwise
                error('please input legal option');
        end
    otherwise
        error('you have input too many parameters');
end

