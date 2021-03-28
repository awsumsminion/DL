clear all; close all;

%% vanilla RNN

% 0.1 Read in data
book_fname = 'goblet_book.txt'; 
fid = fopen(book_fname,'r'); 
book_data = fscanf(fid,'%c'); 
fclose(fid);

book_chars = unique(book_data);
K = size(book_chars, 2);

char_to_ind = containers.Map('KeyType','char','ValueType','int32'); 
ind_to_char = containers.Map('KeyType','int32','ValueType','char');

for i = 1:K
    char_to_ind(book_chars(i)) = i;
    ind_to_char(i) = book_chars(i);
end

% 0.2 Set hyper-params & initialize RNN params
m = 5; % hidden state
eta = 0.1; % learning rate
seq_length = 25; % input sequence length
sig = 0.01;

RNN.b = zeros(m,1);
RNN.c = zeros(K,1);
RNN.U = randn(m,K)*sig;
RNN.W = randn(m,m)*sig;
RNN.V = randn(K,m)*sig;

%% 0.4 Compute gradients
X_chars = book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1);
X = zeros(K, seq_length);
Y = zeros(K, seq_length);
h0 = zeros(m, 1);

for i = 1:seq_length
    x_idx = char_to_ind(X_chars(i)); 
    X(x_idx, i) = 1;
    y_idx = char_to_ind(Y_chars(i)); 
    Y(y_idx, i) = 1;
end

[grads, ~] = ComputeGradients(X, Y, RNN, h0);

h = 1e-4;
grads_num = ComputeGradsNum(X, Y, RNN, h);

%relative error
error_b = abs(norm(grads.b)-norm(grads_num.b)) ./ max(eps, norm(abs(grads.b))+norm(abs(grads_num.b)));
error_c = abs(norm(grads.c)-norm(grads_num.c)) ./ max(eps, norm(abs(grads.c))+norm(abs(grads_num.c)));
error_U = abs(norm(grads.U)-norm(grads_num.U)) ./ max(eps, norm(abs(grads.U))+norm(abs(grads_num.U)));
error_W = abs(norm(grads.W)-norm(grads_num.W)) ./ max(eps, norm(abs(grads.W))+norm(abs(grads_num.W)));
error_V = abs(norm(grads.V)-norm(grads_num.V)) ./ max(eps, norm(abs(grads.V))+norm(abs(grads_num.V)));

fprintf('Error b:'); disp(error_b)
fprintf('Error c:'); disp(error_c)
fprintf('Error U:'); disp(error_U)
fprintf('Error W:'); disp(error_W)
fprintf('Error V:'); disp(error_V)

%avoid exploding gradients
for f = fieldnames(grads)'
    grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
end

%% 0.5 Train your RNN using AdaGrad

b = RNN.b; c = RNN.c; U = RNN.U; W = RNN.W; V = RNN.V;
K = size(c,1); n = size(b,1);
smooth_loss = 0;
e = 1;
e_count = 0;
n_epochs = 2;
N = n_epochs*(length(book_data)-seq_length);
%N = 10000;

for t = 1:N
    m = 100;
    X_chars = book_data(e:e+seq_length-1);
    Y_chars = book_data(e+1:e+seq_length);
    X = zeros(K, seq_length);
    Y = zeros(K, seq_length);
    nb = seq_length;

    for i = 1:seq_length
        x_idx = char_to_ind(X_chars(i)); 
        X(x_idx, i) = 1;
        y_idx = char_to_ind(Y_chars(i)); 
        Y(y_idx, i) = 1;
    end 

    if e == 1
        h = zeros(n, 1);
    else
        h = hprev(:,nb);
    end

    [grads, hprev, loss] = ComputeGradients(X, Y, RNN, h);
    smooth_loss = 0.999*smooth_loss + 0.001*loss;
    loss_vec(t) = smooth_loss;
    
    for f = fieldnames(grads)' % avoid exploding gradients
        grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
    end
    
    for f = fieldnames(RNN)' % AdaGrad
        m = m + sum(sum(grads.(f{1}).^2));
        RNN.(f{1}) = RNN.(f{1}) - eta*grads.(f{1})*(1/(sqrt(m+eps)));
    end
    
    if smooth_loss <= min(min(loss_vec(5000:end)))
        RNN_star = RNN;
        h0_star = hprev(:,nb);
        x0_star = X(:,1);
        loss_star = smooth_loss;
    end
    
    if mod(t,500) == 1
        disp('------------------------------------------------------')
        disp(['Update step: ' num2str(t) ', smooth loss: ' num2str(smooth_loss)])
        x0 = X(:,1);
        h0 = hprev(:,nb);
        [y, ~] = SynSeq(RNN, h0, x0, 200);
        for ind = 1:length(y)
           msg(ind) = ind_to_char(y(ind));
        end
        disp(msg)
    end
    
    e = e + seq_length;
    if e > length(book_data)-seq_length-1
        e = 1;
        e_count = e_count + 1;
        disp('**************************')
        disp(['Epoch: ' num2str(e_count)])
        disp('**************************')
        if e_count == 3
            break
        end
    end
end

x = 1:t;
plot(x, loss_vec)
title('Loss function')
xlabel('update steps')
ylabel('loss')

disp(' ')
disp('-----------------------')
disp('Best model: ')
disp(['Smooth loss: ' num2str(loss_star)])

[y, ~] = SynSeq(RNN_star, h0_star, x0_star, 1000);
for ind = 1:length(y)
   msg(ind) = ind_to_char(y(ind));
end
disp(msg)

%% Functions

function [y, loss] = SynSeq(RNN, h0, x0, n)
    b = RNN.b; c = RNN.c; U = RNN.U; W = RNN.W; V = RNN.V;
    %h{1} = h0; 
    x{1} = x0;
    h(:,1)=h0;
    d = size(U,2); m = size(b,1);
    
    for t = 1:n
        a{t} = W*h(:,t)+U*x{t}+b;
        h(:,t+1) = tanh(a{t});
        o{t} = V*h(:,t+1)+c; % Kxn
        
        P(:,t) = softmax(o{t}); % Kxn
        cp = cumsum(P(:, t));
        
        aa = rand;
        ixs = find(cp-aa >0);
        ii = ixs(1);
        y(t) = ii;
        %x{t+1} = ii*ones(d, 1);
        x{t+1} = zeros(d,1);
        x{t+1}(ii) = 1;
    end
    
    Y = zeros(d,n);
    for i = 1:n
        Y(y(i), i)=1; % Kxn
    end
    
    l_cross = -log(Y'*P);
    loss = (1/n)*(trace(l_cross));
end

function [grads, h, loss] = ComputeGradients(X, Y, RNN, h0)
    b = RNN.b; c = RNN.c; U = RNN.U; W = RNN.W; V = RNN.V;
    nb = size(X,2); n = size(b,1); K = size(c,1);
    
    h = zeros(n,nb+1);
    h(:,1) = h0; 
    
    % forward pass
    for t = 1:nb
        a{t} = W*h(:,t)+U*X(:,t)+b;
        h(:,t+1) = tanh(a{t}); % nx(nb+1)
        o{t} = V*h(:,t+1)+c; % Kxnb
        P(:,t) = softmax(o{t}); % Kxnb   
    end
    
    l_cross = -log(Y'*P);
    loss = (trace(l_cross));
    
    % backprop
    G = -(Y-P); % Kxnb
    grads_l = ones(1,nb);
    grads_p = -((Y'*P).^-1)*Y'; % nbxK
    grads_o = G'; % nbxK
    
    grads_h = zeros(nb,n);
    grads_a = zeros(nb,n);
    grads_h(nb,:) = grads_o(nb,:)*V; % (1xK)(Kxn)=(1xn)
    grads_a(nb,:) = grads_h(nb,:)*diag(1-(tanh(a{nb}).^2)); % nbxn
    
    t = nb-1;
    for i = 1:nb-1
        grads_h(t,:) = grads_o(t,:)*V + grads_a(t+1,:)*W; % (1xK)(Kxn)+(1xn)(nxn)=(1xn)
        grads_a(t,:) = grads_h(t,:)*diag(1-tanh(a{t}).^2);
        t = t-1;
    end
    
    grads.b = grads_h'*ones(nb,1);
    grads.c = G*ones(nb,1);
    grads.U = grads_a'*X';
    grads.W = grads_a'*h(:,1:nb)';
    grads.V = grads_o'*h(:,2:nb+1)'; % (Kxnb) (nb)xn = Kxn
end

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