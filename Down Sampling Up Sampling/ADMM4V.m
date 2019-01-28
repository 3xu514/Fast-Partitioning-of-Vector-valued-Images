function v = ADMM4V(f, gamma, tao, mu0) %in paper, gamma = 0.5 or 2.0, tao = 2, mu0 = 0.01*gamma
    v = f;
    [m,n,s] = size(f);
    mu = mu0;
    stop=sum(sum(sum(f.^2)))*1e-10;
    lambda = zeros(m,n,s);
    u = f;
    error = +Inf;
    while error > stop
        f1 = f+mu*v-lambda;
        u = UPVP_zjq(permute(f1,[2,1,3])/(1+mu), 2*gamma/(1+mu));
        u = permute(u,[2,1,3]);
        
        f2 =f+mu*u+lambda;
        v = UPVP_zjq(f2/(1+mu), 2*gamma/(1+mu));
        lambda = lambda + mu*(u-v);
        mu = tao*mu;
        error = immse(u,v)
    end
end