function [error, result] = ADMM8VN1(f, gamma, mu0, Mu0) %in paper, gamma = 0.5 or 2.0, tau = 2, mu0 = 0.01*gamma
[M,N,S] = size(f);
    mu = mu0;
    Mu = Mu0;
    
    u1 = zeros(M,N,S);
    u2 = zeros(M,N,S);
    u3 = zeros(M,N,S);
    u4 = zeros(M,N,S);
    l1 = zeros(M,N,S);
    l2 = zeros(M,N,S);
    l3 = zeros(M,N,S);
    l4 = zeros(M,N,S);
    o12 = zeros(M,N,S);
    o13 = o12;
    o14 = o12;
    o23 = o12;
    o24 = o12;
    o34 = o12;
    v = sign(-f).*max(abs(f)-1/(4*mu),0)+f;
    
    error = +Inf;
    wc = sqrt(2)-1;
    wd = 1-sqrt(2)/2;
    
    diag_coordinates = zeros(M*N,2); 
    antidiag_coordinates = zeros(M*N,2);
    start_point = zeros(M+N,1);
    TOL = 1e-13*sum(sum(sum(f.^2)))*6;

    counter = 1;
    for i = 1:M+N-1 
        start_point(i) = counter;
        for j = 1:min([M, N, i, M+N-i]) 
            diag_coordinates(counter,1) = max(i-N+1, 1) + j-1;
            diag_coordinates(counter,2) = i - diag_coordinates(counter,1) + 1;
            counter = counter + 1;
        end
    end
    start_point(end) = start_point(end-1) + 1;
    antidiag_coordinates(:,1) = M+1-diag_coordinates(:,1);
    antidiag_coordinates(:,2) = diag_coordinates(:,2);
    
    iter = 1;
    while error > TOL
        % cols
        w1 = (mu*v+l1+(Mu*u2-o12)+(Mu*u3-o13)+(Mu*u4-o14))/(mu+3*Mu);
        u1 = UPVP(permute(w1,[2,1,3]),2*gamma*wc/(mu+3*Mu));
        u1 = permute(u1,[2,1,3]);
        
        %diagonal
        w2 = (mu*v+l2+(Mu*u1+o12)+(Mu*u3-o23)+(Mu*u4-o24))/(mu+3*Mu);
        for i = 1:M+N-1 
            segment = start_point(i):start_point(i+1)-1;    
            for s = 1:S
                temp_s = squeeze(w2(:,:,s));
                temp(i,1:size(segment,2),s) = temp_s(sub2ind([M,N], diag_coordinates(segment,1), diag_coordinates(segment,2)));
            end
        end
        temp = UPVP(temp, 2*gamma*wd/(mu+3*Mu));
        for s = 1:S   
            for i = 1:M+N-1
                segment = start_point(i):start_point(i+1)-1;
                temp_s(sub2ind([M,N], diag_coordinates(segment,1), diag_coordinates(segment,2))) = temp(i,1:size(segment,2),s);     
            end
            u2(:,:,s) = temp_s;
        end
        
        %row
        w3 = (mu*v+l3+(Mu*u1+o13)+(Mu*u2+o23)+(Mu*u4-o34))/(mu+3*Mu);
        u3 = UPVP(w3,2*gamma*wc/(mu+3*Mu));

        w4 = (mu*v+l4+(Mu*u1+o14)+(Mu*u2+o24)+(Mu*u3+o34))/(mu+3*Mu);
        for i = 1:M+N-1 
            segment = start_point(i):start_point(i+1)-1;    
            for s = 1:S
                temp_s = squeeze(w4(:,:,s));
                temp(i,1:size(segment,2),s) = temp_s(sub2ind([M,N], antidiag_coordinates(segment,1), antidiag_coordinates(segment,2)));
            end
        end
        temp = UPVP(temp, 2*gamma*wd/(mu+3*Mu));
        for s = 1:S   
            for i = 1:M+N-1
                segment = start_point(i):start_point(i+1)-1;
                temp_s(sub2ind([M,N], antidiag_coordinates(segment,1), antidiag_coordinates(segment,2))) = temp(i,1:size(segment,2),s);     
            end
            u4(:,:,s) = temp_s;
        end
        Z = (1/4)*((u1-(l1/mu))+(u2-(l2/mu))+(u3-(l3/mu))+(u4-(l4/mu)));
        %v = (f+mu*4*Z)/(1+mu*4);
        v = sign(Z-f).*max(abs(Z-f)-1/(4*mu),0)+f;
        
        l1 = l1 + mu*(v-u1);
        l2 = l2 + mu*(v-u2);
        l3 = l3 + mu*(v-u3);
        l4 = l4 + mu*(v-u4);
        
        o12 = o12+Mu*(u1-u2);
        o13 = o13+Mu*(u1-u3);
        o14 = o14+Mu*(u1-u4);
        o23 = o23+Mu*(u2-u3);
        o24 = o24+Mu*(u2-u4);
        o34 = o34+Mu*(u3-u4);
        
        mu = mu*2^iter;
        Mu = Mu*2^iter;
        
        error(iter) = immse(u1,v)+immse(u2,v)+immse(u3,v)+immse(u4,v);
        error(iter);
        iter = iter+1;
    end
    result = round(u1,3);
end