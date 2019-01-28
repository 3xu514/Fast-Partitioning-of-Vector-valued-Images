function h = UnivPotts(f, gamma, w)
    % --------------------------------------------------------------------
    % This function finds the solution to the univariate Potts problem.
    %
    % [Input]   f: the data vector f \in R^n
    %
    %           gamma: model parameter. small gamma results in detail
    %           partitions; large gamma results in partitions having fewer
    %           partitioning regions.
    %
    %           w: weights w \in (R+)^n
    %
    % [Output]  h: the resulting global minimizer to the univariate Potts
    %           problem.
    % --------------------------------------------------------------------
    N = length(f);
    M = zeros(N+1, 1);  % indices are 0, 1, 2, ...
    S = zeros(N+1, 1);  % indices are 0, 1, 2, ...
    W = zeros(N+1, 1);  % indices are 0, 1, 2, ...
    P = zeros(N, 1);    % indices are 1, 2, 3, ...
    J = zeros(N, 1);    % indices are 1, 2, 3, ...
    %              w:   % indices are 1, 2, 3, ...
    %          gamma:   % single value
    %              f:   % indices are 1, 2, 3, ...
    h = zeros(N, 1);
    % --------------------------------------------------------------------
    % Find the optimal jump locations
    for r = 1:N
        M(r+1) = M(r) + w(r) * f(r);
        S(r+1) = S(r) + w(r) * f(r)^2;
        W(r+1) = W(r) + w(r);
        P(r) = S(r+1) - M(r+1)^2 / W(r+1);
        J(r) = 0;
        
        if(r >= 2) % skip the case where r = 1
            for l = r:-1:2
                d = S(r+1) - S(l) - (M(r+1) - M(l))^2/(W(r+1) - W(l));
                if (P(r) < d + gamma)
                    break;
                end
                p = P(l-1) + gamma + d;
                if(p <= P(r))
                    P(r) = p;
                    J(r) = l-1;
                end
            end
        end
    end
    
    % Reconstruct the minimizer h from the optimal jump locations
    r = N; l = J(r);
    while(r > 0)
        l = J(r);
        for i = l+1:r
            h(i) = sum(w(l+1:r) .* f(l+1:r)) / sum(w(l+1:r));
        end
        r = l;
    end
end