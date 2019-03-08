function [est,esthat] = RMTCovDistEst(X,Y,distance)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
n1=size(X,2);
n2=size(Y,2);
p=size(X,1);
switch distance
    case 'Fisher'
        f = @(t) log(t).^2;
    case 'bhattacharrya'
        f = @(t) -1/4*log(t)+1/2*log(1+t)-1/2*log(2);
    case 'KL'
        f = @(t) 1/2*t-1/2*log(t)-1/2;
    case 't'
        f = @(t) t;
    case 'log'
        f = @(t) log(t);
    case 'log(1+st)'
        s = 1;
        f = @(t) log(1+s*t);
    case 'manual'
        f = @(t) log(t);
end    
    c2=p/n2;
    c1=p/n1;                   
    hatC1=1/n1*(X*X');
    hatC2=1/n2*(Y*Y');
    F=hatC1\hatC2;

    lambda=sort(eig(F));
    slambda=sqrt(lambda);
    eta = sort(eig(diag(lambda)-slambda*slambda'/(p-n1)));
    zeta = sort(eig(diag(lambda)-slambda*slambda'/n2));
    m = @(z) mean(1./(lambda*ones(1,length(z))-ones(p,1)*z));        
    phi=@(z) z+c1*z.^2.*m(z);      
    psi=@(z) 1-c2-c2*z.*m(z);
    switch distance
        case 'Fisher'
            M=zeros(p);
                N=zeros(p);                
                for i=1:p
                    M(i,i)=1/(2*lambda(i)^2);
                    N(i,i)=1/lambda(i);                 
                    js=1:p;
                    js(i)=[];
                    for j=js
                        M(i,j)=(-1+lambda(i)/lambda(j)-log(lambda(i)/lambda(j)))/(lambda(i)-lambda(j))^2;                        
                        N(i,j)=log(lambda(i)/lambda(j))/(lambda(i)-lambda(j));
                    end
                end     

                %%% Large p-estimate
                
                est=2*(c1+c2-c1*c2)/(c1*c2)*( (eta-zeta)'*M*(eta-lambda)+(eta-lambda)'*(log((1-c1)*lambda)./lambda) )...
                    -2/p*(eta-zeta)'*N*ones(p,1)+1/p*sum(log((1-c1)*lambda).^2)...
                    -2*(1-c2)/c2*( 1/2*log( (1-c1)*(1-c2) )^2+(eta-zeta)'*(log((1-c1)*lambda)./lambda) );               
                esthat=mean(f(lambda));
        case 'log(1+st)'  
            %%% additional kappa term in negative side
            s=1;
                kappa_p=0;
                kappa_m=-1/(s*(1-c1));
                if c2>1
                   kappa_p=min(lambda(lambda>1e-3));
                   while phi(kappa_m)/psi(kappa_m)>-1/s
                       kappa_m=2*kappa_m;
                   end
                end
                
                while abs(kappa_p-kappa_m)>1e-6*abs(eta(p)-zeta(p))
                    kappa_=(kappa_p+kappa_m)/2;
                    if phi(kappa_)/psi(kappa_)<-1/s
                        kappa_m=kappa_;
                    else
                        kappa_p=kappa_;
                    end
                end
                kappa_0=(kappa_p+kappa_m)/2;  
            est=(c1+c2-c1*c2)/(c1*c2)*log((c1+c2-c1*c2)/(1-c1)/abs(c2-s*c1*kappa_0))+1/c2*log(abs(-s*kappa_0*(1-c1)))+1/p*sum(log(abs(1-lambda/kappa_0)));  
            esthat=mean(f(lambda));
        case 'bhattacharrya'
            est=(-1/4)*RMTCovDistEst(X,Y,'log')+1/2*RMTCovDistEst(X,Y,'log(1+st)')-1/2*log(2);
            esthat=mean(f(lambda));
        case 'KL'
            est=1/2*((1-c1)*mean(lambda)-mean(log(lambda))+(1-c1)/c1*log(1-c1)-(1-c2)/c2*log(1-c2)-1);
            esthat=mean(f(lambda));
        case 't'
            est=(1-c1)*mean(lambda);
            esthat=mean(f(lambda));
        case 'log'
            est=1/p*sum(log(lambda))-(1-c1)/c1*log(1-c1)+(1-c2)/c2*log(1-c2);
            esthat=mean(f(lambda));
        case 'manual'
            est=sum(-f(-c1/c2*lambda)*((c1+c2-c1*c2)/c1/c2-1/p));
            esthat=mean(f(lambda));
        case 'Wasserstein'
            [est,esthat]=RMTWassDist(X,Y);
    end

end

