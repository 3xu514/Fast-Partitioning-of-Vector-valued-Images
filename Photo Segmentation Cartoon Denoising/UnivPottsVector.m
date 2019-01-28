function h = UnivPottsVector(f, gamma, w)
    % --------------------------------------------------------------------
    % This function finds the solution to the univariate Potts problem.
    %
    % [Input]   f: the data vector f \in R^(N by s)
    %
    %           gamma: model parameter. small gamma results in detail
    %           partitions; large gamma results in partitions having fewer
    %           partitioning regions.
    %
    %           w: weights w \in (R+)^N
    %
    % [Output]  h: the resulting global minimizer to the univariate Potts
    %           problem. h \in R^(N by s)
    % --------------------------------------------------------------------
    [N, s] = size(f);
    M = cumsum(w.*f, 1); 
    S = cumsum(w.*f.^2, 1); 
    W = cumsum(w); 
    P = zeros(N, 1); 
    for r = 1:N
        P(r) = sum((S(r,:) - M(r,:).^2 ./ W(r)));
    end
    J = zeros(N, 1);
    h = zeros(N, s);
    % --------------------------------------------------------------------
    % Find the optimal jump locations
    for r = 2:N
        for l = r:-1:2
            d = sum((S(r,:) - S(l-1,:) - (M(r,:) - M(l-1,:)).^2/ ...
                (W(r) - W(l-1))));
            
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
    
    % Reconstruct the minimizer h from the optimal jump locations
    r = N; l = J(r);
    while(l > 0)
        h(l+1:r,:) = repmat((M(r,:)-M(l,:)) / (W(r)-W(l)), r-l, 1);
        r = l;
        l = J(r);
    end
    h(1:r,:) = repmat((M(r,:)) / (W(r)),r,1);
end