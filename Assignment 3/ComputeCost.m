function [c, r] = ComputeCost(X, Y, NetParams, lambda, varargin)
% P (Kxn), W (Kxd)
% Y contains the one-hot representation (Kxn)
% J corresponds to the sum of the loss of the network's predictions
%   for X relative to the ground truth labels and W
    fprintf('Loading ComputeCost... ');
    W = NetParams.W; b = NetParams.b;
    nb = size(Y,2); k = numel(W);
    H{1} = X;
    sumW = sum(sum(W{1}.^2));
    
    for i=1:k-1
        if NetParams.use_bn
            ga = NetParams.gammas{i}; 
            be = NetParams.betas{i};
            
            s = W{i}*H{i}+b{i};
            
            if nargin == 6
                mu = varargin{1}(i);
                v = varargin{2}(i);
            else
                [~,n] = size(s);
                mu = mean(s,2);
                v = var(s, 0, 2);
                v = v * (n - 1) / n;
            end

            s_hat = (diag((v+eps).^-0.5))*(s-mu);
            s_tilde = ga.*s_hat+be;
            H{i+1} = max(0, s_tilde);
        else
            s = W{i}*H{i}+b{i}*ones(1,nb);
            H{i+1} = max(s,0);
        end
        sumW = sumW + sum(sum(W{i+1}.^2));
    end
    P = softmax(W{k}*H{k}+b{k}*ones(1,nb));
    
    l_cross = -log(Y'*P); % (nxK)*(Kxn) = (nxn)
    
    c = (1/nb)*(trace(l_cross)) + lambda*sumW;
    
    r = (1/nb)*(trace(l_cross));
    
    disp('Done!');
end