clear all; close all; clc;
addpath Datasets/cifar-10-batches-mat/;
training = 'data_batch_1.mat';
validation = 'data_batch_2.mat';
testing = 'test_batch.mat';

%% Top Level
K = 10; n = 10000; d = 3072; m = 50;
h = 1e-6; eps = 1e-6; rng('default');
lambda = 0.01;

%training data
[trainX, trainY, trainy] = LoadBatch(training);

trainX = trainX./255;
mean_X = mean(trainX, 2);
std_X = std(trainX, 0, 2);

trainX = trainX - repmat(mean_X, [1, size(trainX, 2)]); 
trainX = trainX ./ repmat(std_X, [1, size(trainX, 2)]); % normalizing training data

% validation data
[valX, valY, valy] = LoadBatch(validation);
valX = valX./255;
valX = valX - repmat(mean_X, [1, size(valX, 2)]); 
valX = valX ./ repmat(std_X, [1, size(valX, 2)]); % normalizing validation data

% test data
[testX, testY, testy] = LoadBatch(testing);
testX = testX./255;
testX = testX - repmat(mean_X, [1, size(testX, 2)]); 
testX = testX ./ repmat(std_X, [1, size(testX, 2)]); % normalizing validation data

[W, b] = ParameterasInit(d, m, K); % {W1, W2, b1, b2}


%% Cyclical learning rates
eta_min = 1e-5;
eta_max = 1e-1;
n_s = 800; % stepsize k(n/n_batch) --> 5*10000/100 = 500
n_batch = 100; 
num_cycles = 3;
n_epochs = (num_cycles*2*n_s*n_batch)/n;
lambda = 0.01;
l = 0; % cycles l = 0, 1, 2, 3, ...
t_min = 2*l*n_s;
t_mid = (2*l+1)*n_s;
t_max = 2*(l+1)*n_s;
t = 1;

Jtrain = zeros(1, 2*n_s);
Wstar = W; bstar = b;
 
for i = 1:n_epochs
    for j=1:n/n_batch % generates mini-batches for one epoch
            if (t_min <= t) && (t <= t_mid)
                eta_t = eta_min + (t - (2*l*n_s))*(eta_max - eta_min)/n_s;
            end
            if (t_mid <= t) && (t <= t_max)
                eta_t = eta_max - (t - ((2*l+1)*n_s))*(eta_max - eta_min)/n_s;
            end
            
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            XbatchT = trainX(:, j_start:j_end); % (dxnb)
            YbatchT = trainY(:, j_start:j_end); % (Kxnb) 
            ybatchT = trainy(j_start:j_end);
            [Wstar, bstar] = MiniBatchGD(XbatchT, YbatchT, eta_t, Wstar, bstar, lambda);
            [c, r] = ComputeCost(XbatchT, YbatchT, Wstar, bstar, lambda);
            Ptrain = EvaluateClassifier(XbatchT, Wstar, bstar);
            acctrain(t) = ComputeAccuracy(Ptrain,ybatchT); Jtrain(t) = c; losstrain(t) = r;
            
            XbatchV = valX(:, j_start:j_end); % (dxnb)
            YbatchV = valY(:, j_start:j_end); % (Kxnb) 
            ybatchV = valy(j_start:j_end);
            [c, r] = ComputeCost(XbatchV, YbatchV, Wstar, bstar, lambda);
            Pval = EvaluateClassifier(XbatchV, Wstar, bstar);
            accval(t) = ComputeAccuracy(Pval,ybatchV); Jval(t) = c; lossval(t) = r;

            t = t + 1;
            if t > t_max
                l = l + 1;
                t_min = 2*l*n_s;
                t_mid = (2*l+1)*n_s;
                t_max = 2*(l+1)*n_s;
            end
            
    end
end
 
Pstar = EvaluateClassifier(testX, Wstar, bstar);
testacc = ComputeAccuracy(Pstar,testy);
x1 = 1:t-1;
x2 = 1:t-1;
%x = n_epochs;
 
figure(1)
subplot(1,3,1)
plot(x1(1:q:end), Jtrain(1:q:end), x2(1:q:end), Jval(1:q:end))
title('Cost function')
xlabel('update steps')
ylabel('cost')
legend('training')

subplot(1,3,2)
plot(x1(1:q:end), losstrain(1:q:end), x2(1:q:end), lossval(1:q:end))
title('Loss function')
xlabel('update steps')
ylabel('loss')
legend('training', 'validation')

subplot(1,3,3)
plot(x1(1:q:end), acctrain(1:q:end), x2(1:q:end), accval(1:q:end))
title('Accuracy')
xlabel('update steps')
ylabel('accuracy')
legend('training', 'validation')


%% Coarse search
%{
format long;
K = size(trainY,1); % 10
d = size(trainX, 1); % 3072
m = 50;

ntrain = size(trainX, 2); % 45000
nval = size(valX, 2); % 5000

eta_min = 1e-5;
eta_max = 1e-1;
n_batch = 100; 
num_cycles = 2;

n_s = 2*floor(ntrain/n_batch);
n_epochs = (num_cycles*2*n_s*n_batch)/ntrain;

lmin = -4.2; lmax = -2.5;
for i = 1:8
    l(i) = lmin + (lmax - lmin)*rand(1, 1); 
    lambda(i) = 10^l(i);
end

GDparams = [n_batch, eta_min, eta_max, n_epochs, n_s];

for i = 1:length(lambda)
    [Wstar, bstar] = ParameterasInit(d, m, K); % {W1, W2, b1, b2}
    acc(i) = FindLambda(trainX, trainY, valX, valy, Wstar, bstar, GDparams, lambda(i));
end

result = [l;lambda;acc]
%}

%% Functions

function [X, Y, y] = LoadBatch(fname) 
% Updated!
    fprintf('Loading LoadBatch... ');
    K = 10;
    A = load(fname);
    X = transpose(A.data);
    X = cast(X,'double');
    n = size(X, 2);
    y = A.labels + 1;
    y = cast(y,'double');
    Y = zeros(K, n);
    for i = 1:n
        onehot = y(i);
        Y(onehot, i) = 1;
    end
    disp('Done!');
end

function [W, b] = ParameterasInit(d, m, K)
    dev1 = 1/sqrt(d); dev2 = 1/sqrt(m);
    W1 = dev1.*randn(m,d);
    W2 = dev2.*randn(K,m);
    b1 = zeros(m,1); % b1 = dev.*randn(m,1) + mean_value;
    b2 = zeros(K,1); % b2 = dev.*randn(K,1) + mean_value;
    W = {W1, W2};
    b = {b1, b2};
end

function P = EvaluateClassifier(X, W, b)
% Updated!
    fprintf('Loading EvaluateClassifier... ');
    W1 = W{1}; W2 = W{2}; b1 = b{1}; b2 = b{2};
    n = size(X,2);
    H = max(W1*X+b1*ones(1,n),0);
    P = softmax(W2*H+b2*ones(1,n));
    disp('Done!');
end

function acc = ComputeAccuracy(P, y)
% No changes!
    fprintf('Loading ComputeAccuracy... ');
    [p_max, index] = max(P);
    index = index'; 
    a = 0;
    for i = 1:size(index)
        if index(i) == y(i)
            a = a + 1;
        end
    end
    acc = a/size(index,1);
    disp('Done!');
end

function [grad_b, grad_W] = ComputeGradients(X, Y, W, b, lambda)
% X data (dxn), Y one-hot (Kxn), P probability (Kxn), W (Kxd)
% grad_W is the gradient matrix of the cost J relative to W (Kxd)
% grad_b is the gradient vector of the cost J relative to b (Kx1)
    nb = size(Y,2);
    W1 = W{1}; W2 = W{2}; b1 = b{1}; b2 = b{2};
    
    % forward pass
    H = max(W1*X+b1*ones(1,nb),0);
    P = softmax(W2*H+b2*ones(1,nb));
    
    % backward pass
    G = -(Y-P);
    grad_W2 = (1/nb)*G*H'+2*lambda*W2;
    grad_b2 = (1/nb)*G*ones(nb,1);
    
    G = W2'*G;
    G = G.*sign(H);
    grad_W1 = (1/nb)*G*X'+2*lambda*W1;
    grad_b1 = (1/nb)*G*ones(nb,1);
    
    grad_W = {grad_W1, grad_W2};
    grad_b = {grad_b1, grad_b2};
end


function [Wstar, bstar] = MiniBatchGD(X, Y, eta, W, b, lambda)
% X training images (dxn), Y labels (Kxn)
% W (Kxd) and b (Kx1) initial values, GDparams = [n_batch, eta, n_epochs]
    [grad_b, grad_W] = ComputeGradients(X, Y, W, b, lambda);
    Wstar1 = W{1} - eta*grad_W{1}; % (Kxd)
    Wstar2 = W{2} - eta*grad_W{2}; % (Kxd)
    
    bstar1 = b{1} - eta*grad_b{1}; % (Kx1)
    bstar2 = b{2} - eta*grad_b{2}; % (Kx1)
    Wstar = {Wstar1, Wstar2};
    bstar = {bstar1, bstar2};
end

function val_acc = FindLambda(trainX, trainY, valX, valy, Wstar, bstar, GDparams, lambda)
    n_batch = GDparams(1);
    eta_min = GDparams(2);
    eta_max = GDparams(3);
    n_epochs = GDparams(4);
    n_s = GDparams(5); % GDparams = [n_batch, eta_min, eta_max, n_epochs, n_s];
    
    n = size(trainX, 2);
    
    l = 0;
    t_min = 2*l*n_s;
    t_mid = (2*l+1)*n_s;
    t_max = 2*(l+1)*n_s;
    t = 1;
    
    for i = 1:n_epochs
        for j=1:n/n_batch % generates mini-batches for one epoch
                if (t_min <= t) && (t <= t_mid)
                    eta_t = eta_min + (t - (2*l*n_s))*(eta_max - eta_min)/n_s;
                end
                if (t_mid <= t) && (t <= t_max)
                    eta_t = eta_max - (t - ((2*l+1)*n_s))*(eta_max - eta_min)/n_s;
                end

                j_start = (j-1)*n_batch + 1;
                j_end = j*n_batch;

                XbatchT = trainX(:, j_start:j_end); % (dxnb)
                YbatchT = trainY(:, j_start:j_end); % (Kxnb) 
                
                [Wstar, bstar] = MiniBatchGD(XbatchT, YbatchT, eta_t, Wstar, bstar, lambda);

                t = t + 1;
                if t > t_max
                    l = l + 1;
                    t_min = 2*l*n_s;
                    t_mid = (2*l+1)*n_s;
                    t_max = 2*(l+1)*n_s;
                end

        end
    end

    Pval = EvaluateClassifier(valX, Wstar, bstar);
    val_acc = ComputeAccuracy(Pval,valy);
end

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
    
    c = (1/n)*(trace(l_cross)) + lambda*(sum(sum(W1.^2))+sum(sum(W2.^2)));
    
    r = (1/n)*(trace(l_cross));
    
    disp('Done!');
end
