function [c, r] = ComputeCost(X, Y, W, b, lambda)
% P (Kxn), W (Kxd)
% Y contains the one-hot representation (Kxn)
% J corresponds to the sum of the loss of the network's predictions
%   for X relative to the ground truth labels and W
    fprintf('Loading ComputeCost... ');
    n = size(Y,2); % n
    W1 = W{1}; W2 = W{2}; b1 = b{1}; b2 = b{2};
    
    H = max(W1*X+b1*ones(1,n),0); % (mxd)*(dxn) + (mx1)*(1xn) = (mxn)
    %H = H(1:db, nb); % dbx1
    P = softmax(W2*H+b2*ones(1,n)); % (Kxm)*(mxn)+(Kx1)*(1xn) = (Kxn)
    
    l_cross = -log(Y'*P); % (nxK)*(Kxn) = (nxn)
    
    c = (1/n)*(trace(l_cross)) + lambda*(sum(sum(W1.^2))+sum(sum(W2.^2))); % cost
    
    r = (1/n)*(trace(l_cross)); % loss
    
    disp('Done!');
end