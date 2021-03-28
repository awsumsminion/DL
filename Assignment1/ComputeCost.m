function J = ComputeCost(X, Y, W, b, lambda)
% P (Kxn), W (Kxd)
% Y contains the one-hot representation (Kxn)
% J corresponds to the sum of the loss of the network's predictions
%   for X relative to the ground truth labels and W
    fprintf('Loading ComputeCost... ');
    K = size(W,1); n = size(X,2);
    for i = 1:n
        s = W*X(:,i)+b; % eq 1 (Kxd)*(dx1)+(Kx1)=(Kx1)
        P(:,i) = softmax(s); % (Kxn)
    end

    D = size(P,2);
    l_cross = -log(Y'*P); % (nxK)*(Kxn) = (nxn)
    J = (1/D)*(trace(l_cross))+lambda*sum(W.^2, 'all');
    disp('Done!');
end