function [ v,lambda,mu ] = DownS_pre(f, gamma, tao, mu0)
%DOWNSPRE Summary of this function goes here
%   Detailed explanation goes here
%   First step in ADMM4
%   same inputs as ADMM4
%   
%% downsampling
    [M,N,S] = size(f);
    
    m=mod(1:M,2)==1;
    n=mod(1:N,2)==1;
    f0=f(m,n,:);
    %% ADMM4
    v0 = f0;
    mu = mu0/2;
    stop=sum(sum(sum(f0.^2)))*1e-8;
    lambda0 = zeros(size(f0));
    u = zeros(size(f0));
    error = +Inf;
    gamma=gamma/2;
    while error>stop
        f1 = f0+mu*v0-lambda0;
        u = UPVP_zjq(permute(f1,[2,1,3])/(1+mu), 2*gamma/(1+mu));
        u = permute(u,[2,1,3]);
        
        f2 = f0+mu*u+lambda0;
        v0 = UPVP_zjq(f2/(1+mu), 2*gamma/(1+mu));
        lambda0 = lambda0 + mu*(u-v0);
        mu = tao*mu;
        error = immse(u,v0)
    end
    %% resize back
    v=[];
    lambda=[];
    for i=1:S
        v(:,:,i) = kron(v0(:,:,i),ones(2,2));
        lambda(:,:,i) = kron(lambda0(:,:,i),ones(2,2));
    end
    v=v(1:M,1:N,:);
    lambda=lambda(1:M,1:N,:);
    mu=mu*2;
end

