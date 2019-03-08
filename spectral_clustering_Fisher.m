clear all;

type = 'synthetic';
%type = 'real';

method = 'approx'; %%% (among 'approx','integral','exact')

switch type
    case 'synthetic'
        p = 128;
        ms = [100 100];
        m = sum(ms);
        
        %ns = [256*ones(1,m/4) 512*ones(1,m/4) 256*ones(1,m/4) 512*ones(1,m/4)];
        %ns = floor((2*p)+rand(1,m)*(2*p));
        ns = [512*ones(1,m-1) 256];
        n = max(ns);
        
        k = length(ms);
        
        C = zeros(p,p,k); 

        X = zeros(p,n,m);
        hatC = zeros(p,p,m);
        
        toeplitz_coeffs=[0,.05];
        
        for i=1:k
            C(:,:,i) = toeplitz(toeplitz_coeffs(i).^(0:p-1));
            %C(:,:,i) = Crand+.025*(i-1)*eye(p);
            sqrtC = (reshape(C(:,:,i),p,p))^(1/2);
            for j=1:ms(i)
                X(:,:,j+sum(ms(1:i-1))) = [sqrtC*randn(p,ns(j+sum(ms(1:i-1)))) zeros(p,n-ns(j+sum(ms(1:i-1))))];
                hatC(:,:,j+sum(ms(1:i-1))) = X(:,:,j+sum(ms(1:i-1)))*X(:,:,j+sum(ms(1:i-1)))'/ns(j+sum(ms(1:i-1)));
            end
        end
        
    case 'real'
        p=64;
        ms=[100 100];
        m=sum(ms);
        k=2;
        ns = 1537*ones(1,m);
        
        hatC = zeros(p,p,m);
        
        fileInfo = dir('*.mat');            
        
        individu = 2;
        mat1 = open(fileInfo(individu).name);
        
        hatC_tmp = permute(mat1.C,[2 3 1]);
        hatC(:,:,1:100)   = hatC_tmp(:,:,mat1.y==0);
        hatC(:,:,101:200) = hatC_tmp(:,:,mat1.y==1);
        
end

f = @(t) exp(-t/2);
K = zeros(m);
K_naiv = zeros(m);

for i=1:m
    for j=i+1:m
        n1 = ns(i);
        n2 = ns(j);
        
        c1 = p/n1;
        c2 = p/n2;
                       
        lambda = sort(eig(hatC(:,:,i)\(hatC(:,:,j))),'ascend');
        eta = sort(eig(diag(lambda)+1/(n1-p)*sqrt(lambda)*sqrt(lambda)'),'ascend');
        zeta = sort(eig(diag(lambda)-1/n2*sqrt(lambda)*sqrt(lambda)'),'ascend');
        
        switch method
            case 'integral'
                
                %%% Integral formula: fast
                altitude = 1;
                
                step = 1e-3;
                min_lambda=min(lambda);
                max_lambda=max(lambda);
                maxV = max_lambda*1.5;
                minV = min_lambda*.5;
                
                contour = fliplr([altitude*1i+minV+(0:step:maxV-minV),altitude*1i+maxV+(-step:-step:-2)*1i*altitude,-1i*altitude+maxV+(-step:-step:-maxV+minV),minV-1i*altitude+(0:step:2)*1i*altitude]);
                integrand = @(z) log(phi(z)./psi(z)).^2./phi(z)/c2.*(psi(z).*phip(z)-phi(z).*psip(z));
                D_Fisher_square=real(trapz(contour,1/(2*pi*1i)*integrand(contour)));
                
            case 'exact'
                %%% Exact formula: slow
                D_Fisher_square=(c1+c2-c1*c2)/c1/c2*(2*sum(sum(polylog(2,1-(zeta*ones(1,p))./(ones(p,1)*lambda'))-polylog(2,1-(eta*ones(1,p))./(ones(p,1)*lambda'))+polylog(2,1-(eta*ones(1,p))./(ones(p,1)*eta'))-polylog(2,1-(zeta*ones(1,p))./(ones(p,1)*eta'))))+sum(log((1-c1)*eta).^2-log((1-c1)*lambda).^2))...
                    -(1-c2)/c2*(log(1-c2)^2-log(1-c1)^2+sum(log(eta).^2-log(zeta).^2))...
                    -1/p*(2*sum(sum(polylog(2,1-(zeta*ones(1,p))./(ones(p,1)*lambda'))-polylog(2,1-(eta*ones(1,p))./(ones(p,1)*lambda'))))-sum(log((1-c1)*lambda).^2));
                
            case 'approx'
                %%% Approximate formula: fast
                M=zeros(p);
                N=zeros(p);
                for i1=1:p
                    M(i1,i1)=1/(2*lambda(i1)^2);
                    N(i1,i1)=1/lambda(i1);
                    js=1:p;
                    js(i1)=[];
                    for j1=js
                        M(i1,j1)=(-1+lambda(i1)/lambda(j1)-log(lambda(i1)/lambda(j1)))/(lambda(i1)-lambda(j1))^2;
                        N(i1,j1)=log(lambda(i1)/lambda(j1))/(lambda(i1)-lambda(j1));
                    end
                end
                
                D_Fisher_square=2*(c1+c2-c1*c2)/(c1*c2)*( (eta-zeta)'*M*(eta-lambda)+(eta-lambda)'*(log((1-c1)*lambda)./lambda) )...
                    -2/p*(eta-zeta)'*N*ones(p,1)+1/p*sum(log((1-c1)*lambda).^2)...
                    -2*(1-c2)/c2*( 1/2*log( (1-c1)*(1-c2) )^2+(eta-zeta)'*(log((1-c1)*lambda)./lambda) );
                
        end
        
        K(i,j)=f(D_Fisher_square);
        K(j,i)=K(i,j);
        
        K_naiv(i,j) = f(sum(log(lambda).^2)/p);
        K_naiv(j,i) = K_naiv(i,j);
    end
    K(i,i)=f(0);
    K_naiv(i,i)=f(0);
    
    i
end

D = diag(K*ones(m,1));
Lap = 1./sqrt(D)*K*1./sqrt(D);

D_naiv = diag(K_naiv*ones(m,1));
Lap_naiv = 1./sqrt(D_naiv)*K_naiv*1./sqrt(D_naiv);

[U L]=eigs(K,k);
[U_naiv L_naiv]=eigs(K_naiv,k);

figure;
hold on;
plot(U(:,2)*sign(U(1,2)));
plot(U_naiv(:,2)*sign(U_naiv(1,2)),'r');

figure;
hold on;
plot(U(:,1)*sign(U(1,1)));
plot(U_naiv(:,1)*sign(U_naiv(1,1)),'r');

figure;
hold on;
plot(U(1:ms(1),1)*sign(U(1,1)),U(1:ms(1),2)*sign(U(1,2)),'x');
plot(U(ms(1)+1:end,1)*sign(U(1,1)),U(ms(1)+1:end,2)*sign(U(1,2)),'kx');

plot(U_naiv(1:ms(1),1)*sign(U_naiv(1,1)),U_naiv(1:ms(1),2)*sign(U_naiv(1,2)),'or');
plot(U_naiv(ms(1)+1:end,1)*sign(U_naiv(1,1)),U_naiv(ms(1)+1:end,2)*sign(U_naiv(1,2)),'oy');

