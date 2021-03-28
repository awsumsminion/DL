function loss = ComputeLoss(X, Y, RNN, h0)
    b = RNN.b; c = RNN.c; U = RNN.U; W = RNN.W; V = RNN.V;
    nb = size(X,2); n = size(b,1);
    
    h = zeros(n,nb+1);
    h(:,1) = h0; 

    for t = 1:nb
        a{t} = W*h(:,t)+U*X(:,t)+b;
        h(:,t+1) = tanh(a{t}); % nx(n+1)
        o{t} = V*h(:,t+1)+c; % Kxn
        P(:,t) = softmax(o{t}); % Kxn   
    end
    
    l_cross = -log(Y'*P);
    loss = (trace(l_cross));
end