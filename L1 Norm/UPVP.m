function h = UPVP(f, gamma)
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
    %           problem. h \in R^(N by 3)
    % --------------------------------------------------------------------
    [R, N, s] = size(f);
    M = cumsum(f, 2);  % (FirstDim) indices are 0, 1, 2, ...
    S = cumsum(f.^2, 2);  % (FirstDim) indices are 0, 1, 2, ...
    W = cumsum(ones(R,N),2);  % (FirstDim) indices are 0, 1, 2, ...
    P = zeros(R, N);    % indices are 1, 2, 3, ...
    J = zeros(R, N);    % indices are 1, 2, 3, ...
    %              w:   % indices are 1, 2, 3, ...
    %          gamma:   % single value
    %              f:   % indices are 1, 2, 3, ...
    h = zeros(R, N, s);
    % --------------------------------------------------------------------
    % Find the optimal jump locations
    P = sum((S - M.^2 ./ W),3);
    for r = 2:N
        %M(:,r+1,:) = M(:,r,:) + w(:,r) .* f(:,r,:);
        %S(:,r+1,:) = S(:,r,:) + w(:,r) .* f(:,r,:).^2;
        %W(:,r+1) = W(:,r) + w(:,r);
        %P(:,r) = sum((S(:,r,:) - M(:,r,:).^2 ./ r),3);      
        for l = r:-1:2
            d = sum(S(:,r,:) - S(:,l-1,:) - (M(:,r,:) - M(:,l-1,:)).^2./ ...
                    (W(:,r) - W(:,l-1)),3);
            idx = find(P(:,r) >= d + gamma);
            if sum(idx) == 0
                break;
            end
            p = inf*ones(R,1);
            p(idx) = P(idx,l-1) + gamma + d(idx);
            idx = find(p<=P(:,r));
            if sum(idx) ~= 0
                P(idx,r) = p(idx);
                J(idx,r) = l-1;
            end
        end
    end
    
    % Reconstruct the minimizer h from the optimal jump locations
    for i = 1:R
        r = N; l = J(i,r);
        while(l > 0)
            h(i,l+1:r,:) = repmat((M(i,r,:)-M(i,l,:))/(W(i,r)-W(i,l)), r-l, 1);
            r = l;l = J(i,r);
        end
        h(i,1:r,:) = repmat((M(i,r,:))/(W(i,r)), r-l, 1);
    end
end