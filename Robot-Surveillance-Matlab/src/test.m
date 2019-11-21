%this is a test%
n=8;
%%%%%%%%%star graph%%%%%%%
% A=zeros(n,n);
% A(1,:)=1;
% A(:,1)=1;
% A(1,1)=0;
% for i=1:n
%     A(i,i)=round(rand(1,1)*1.4);
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%ring graph%%%%%%%
% A=zeros(8,8);
% for i=2:n-1
%     A(i,i)=round(rand(1,1)*1.4);
%     A(i,i+1)=1;
%     A(i+1,i)=1;
%     A(i-1,i)=1;
%     A(i,i-1)=1;
% end
% A(1,2)=1;A(2,1)=1;A(1,n)=1;A(n,1)=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%complete graph%%%%%
% A=ones(8,8);
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%random graph%%%%%%%
% A=round(rand(8,8));
% A=A+A';
% for i=1:n
%     for j=1:n
%         if A(i,j)==2
%             A(i,j)=1;
%         end
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%line graph%%%%%%%%
% A=zeros(n,n);
% for i=2:n-1
%     A(i,i)=round(rand(1,1)*1.4);
%     A(i,i+1)=1;
%     A(i-1,i)=1;
%     A(i,i-1)=1;
% end
% A(1,2)=1;
% A(n,n-1)=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%
W=zeros(n,n);
for i=1:n
    for j=1:n
        if A(i,j)>0
            W(i,j)=round(rand(1,1)*10)+1;
        end
    end
end
A_inv=ones(n,n);
% A=zeros(n,n);
A_inv(A>0)=0;
% A(W>0)=1;
A_vec=zeros(n^2,1);
A_inv_vec=zeros(n^2,1);
A_inv_vec1=zeros(n^2,1);
A_eq1=zeros(n,n^2);
for i=1:n^2
    A_vec(i)=A(i);
    if A_inv(i)==1
        A_inv_vec(i)=A_inv(i);
        A_eq1=[A_eq1;A_inv_vec'];
        A_inv_vec=zeros(n^2,1);
        A_inv_vec1(i)=1;
        continue
    end       
end
A1=[diag(A_vec);-diag(ones(n^2,1))];
b1=[A_vec;-0.1*A_vec];
for i=1:n
    for j=i:n:n^2
        A_eq1(i,j)=1;
    end
end
b_eq1=[ones(n,1);zeros(size(A_eq1,1)-n,1)];
X0=A./repmat(sum(A,2),[1,n]);
P0=zeros(n^2,1);
P0=X0(:);
fa=rand(size(P0));
xnew = linprog(fa,A1,b1,A_eq1,b_eq1);
v_place=[];
for i = 1:n
    for j = 1:n
       v_place=[v_place j+(i-1)*n];
    end
end
P=zeros(n,n);
P(v_place)=xnew;
tau=10;
yeta=0.1;
epsilon=0.08;
PI=stadis(P);
F_Hit=MC_EVA(P,W,tau,'HittingTime');
F_Entro=MC_EVA(P,'EntropyRate');
F_Kem=MC_EVA(P,W,'Kemeny');
F_Ret=MC_EVA(P,W,yeta,'ReturnTimeEntropy');
F_Mix=MixingTime(P);
F_Hit_Op=MC_OP(A,W,tau,'HittingTimeOp');
F_Kem_Op=MC_OP(A,PI,W,'KemenyOp');
F_Mix_Op=MC_OP(A,'MixingTimeOp');
% F_Mix_Op=full(F_Mix_Op);
F_Ret_Op=MC_OP(A,PI,W,epsilon,yeta,'ReturnTimeEntropyOp');
F_Entro_Op=MC_OP(A,PI,'EntropyRateOp');