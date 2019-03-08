clear all
clc
close all
p=64;n1=1024;n2=2048;
C1=toeplitz(0.2.^(0:p-1));
C2=toeplitz(0.4.^(0:p-1));
n_simulation=10;
%Choose the distance(Fisher,bhattacharrya,KL,Wasserstein)
distance='Fisher';
switch distance
    case 'Fisher'
        f=@(z) log(z).^2;
    case 'bhattacharrya'
        f = @(t) -1/4*log(t)+1/2*log(1+t)-1/2*log(2);
    case 'KL'
        f = @(t) 1/2*t-1/2*log(t)-1/2;
end
for i=1:n_simulation
    X=zeros(p,n1);
    Y=zeros(p,n2);
for k=1:n1
    X(:,k) = mvnrnd(zeros(1,p),C1);
end
for k=1:n2
    Y(:,k) = mvnrnd(zeros(1,p),C2);
end
%Proposed Wasserstein distance
[est(i),esthat(i)] = RMTCovDistEst(X,Y,distance);
%Real Wasserstein distance
if strcmp(distance,'Wasserstein')
    est_vrai(i)=(1/p)*(trace(C1)+trace(C2)-2*trace((C1^(1/2)*C2*C1^(1/2))^(1/2)));
else
    est_vrai(i)=mean(f(eig(C1^(-1)*C2)));
end
end
est_mean=mean(est)
esthat_mean=mean(esthat)
est_vrai_mean=mean(est_vrai)