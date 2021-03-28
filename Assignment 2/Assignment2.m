clear all; close all; clc;
addpath Datasets/cifar-10-batches-mat/;
training = 'data_batch_1.mat';
validation = 'data_batch_2.mat';
testing = 'test_batch.mat';

%% Top Level
% K = 10; n = 10000; d = 3072; m = 50;
% h = 1e-6; eps = 1e-6; %rng(400);

% training data
%[trainX, trainY, trainy] = LoadBatch(training);

[X1, Y1, y1] = LoadBatch('data_batch_1.mat');
[X2, Y2, y2] = LoadBatch('data_batch_2.mat');
[X3, Y3, y3] = LoadBatch('data_batch_3.mat');
[X4, Y4, y4] = LoadBatch('data_batch_4.mat');
[X5, Y5, y5] = LoadBatch('data_batch_5.mat');

trainX = [X1 X2 X3 X4 X5(:, 1:5000)];
trainY = [Y1 Y2 Y3 Y4 Y5(:, 1:5000)];
trainy = [y1 ; y2 ; y3 ; y4 ; y5(1:5000)];

trainX = trainX./255;
mean_X = mean(trainX, 2);
std_X = std(trainX, 0, 2);

trainX = trainX - repmat(mean_X, [1, size(trainX, 2)]); 
trainX = trainX ./ repmat(std_X, [1, size(trainX, 2)]); % normalizing training data

% validation data
%[valX, valY, valy] = LoadBatch(validation);
valX = X5(:, 5001:10000);
valY = Y5(:, 5001:10000);
valy = y5(5001:10000);

valX = valX./255;
valX = valX - repmat(mean_X, [1, size(valX, 2)]); 
valX = valX ./ repmat(std_X, [1, size(valX, 2)]); % normalizing validation data

% test data
[testX, testY, testy] = LoadBatch(testing);
testX = testX./255;
testX = testX - repmat(mean_X, [1, size(testX, 2)]); 
testX = testX ./ repmat(std_X, [1, size(testX, 2)]); % normalizing validation data

%% Gradients
nb = 1; % max n = 10000
db = 20; % max d = 3072
[W, b] = ParameterasInit(d, m, K); % {W1, W2, b1, b2}
aW1 = W{1}(:, 1:db); aW2 = W{2}; aW = {aW1, aW2};
nW1 = W{1}(:, 1:db); nW2 = W{2}; nW = {nW1, nW2};
lambda = 0.01;

% computing gradients analytically
[grad_b, grad_W] = ComputeGradients(trainX(1:db, nb), trainY(:, nb), aW, b, lambda);

% computing gradients numerically 
[ngrad_b, ngrad_W] = ComputeGradsNum(trainX(1:db, nb), trainY(:, nb), nW, b, lambda, 1e-5);

%relative error
grad_W1 = grad_W{1}; grad_W2 = grad_W{2}; grad_b1 = grad_b{1}; grad_b2 = grad_b{2};
ngrad_W1 = ngrad_W{1}; ngrad_W2 = ngrad_W{2}; ngrad_b1 = ngrad_b{1}; ngrad_b2 = ngrad_b{2};

error_W1 = abs(norm(grad_W1)-norm(ngrad_W1)) ./ max(eps, norm(abs(grad_W1))+norm(abs(ngrad_W1)))
error_W2 = abs(norm(grad_W2)-norm(ngrad_W2)) ./ max(eps, norm(abs(grad_W2))+norm(abs(ngrad_W2)))

error_b1 = abs(norm(grad_b1)-norm(ngrad_b1)) ./ max(eps, norm(abs(grad_b1))+norm(abs(ngrad_b1)))
error_b2 = abs(norm(grad_b2)-norm(ngrad_b2)) ./ max(eps, norm(abs(grad_b2))+norm(abs(ngrad_b2)))


%% Cyclical learning rates
K = size(trainY,1); % 10
d = size(trainX, 1); % 3072
m = 50;

n = size(trainX, 2); % 45000
nval = size(valX, 2); % 5000

eta_min = 1e-5;
eta_max = 1e-1;
n_batch = 100; 
num_cycles = 3;
n_s = 2*floor(n/n_batch);
n_epochs = (num_cycles*2*n_s*n_batch)/n;

lambda = 0.000208516905834; % best lambda
l = 0;
t_min = 2*l*n_s;
t_mid = (2*l+1)*n_s;
t_max = 2*(l+1)*n_s;
t = 1;

[Wstar, bstar] = ParameterasInit(d, m, K);

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

        % training 
        XbatchT = trainX(:, j_start:j_end); % (dxnb)
        YbatchT = trainY(:, j_start:j_end); % (Kxnb) 
        ybatchT = trainy(j_start:j_end);

        [Wstar, bstar] = MiniBatchGD(XbatchT, YbatchT, eta_t, Wstar, bstar, lambda);

        if mod(t,10) == 1
            [c, r] = ComputeCost(XbatchT, YbatchT, Wstar, bstar, lambda);
            Ptrain = EvaluateClassifier(XbatchT, Wstar, bstar);
            acctrain(t) = ComputeAccuracy(Ptrain,ybatchT); Jtrain(t) = c; losstrain(t) = r;
        end

        %validation
        XbatchV = valX; % (dxnb)
        YbatchV = valY; % (Kxnb) 
        ybatchV = valy;

        if mod(t,10) == 1
            [c, r] = ComputeCost(XbatchV, YbatchV, Wstar, bstar, lambda);
            Pval = EvaluateClassifier(XbatchV, Wstar, bstar);
            accval(t) = ComputeAccuracy(Pval,ybatchV); Jval(t) = c; lossval(t) = r;
        end

        t = t + 1;
        if t > t_max
            l = l + 1;
            t_min = 2*l*n_s;
            t_mid = (2*l+1)*n_s;
            t_max = 2*(l+1)*n_s;
        end
            
    end
end
x = 1:t-1;

Pstar = EvaluateClassifier(testX, Wstar, bstar);
testacc = ComputeAccuracy(Pstar,testy);

disp(['Test accuracy: ' num2str(testacc)]);

figure(1)
subplot(1,3,1)
plot(x(1:10:end), Jtrain(1:10:end), x(1:10:end), Jval(1:10:end))
title('Cost function')
xlabel('update steps')
ylabel('cost')
legend('training', 'validation')

subplot(1,3,2)
plot(x(1:10:end), losstrain(1:10:end), x(1:10:end), lossval(1:10:end))
title('Loss function')
xlabel('update steps')
ylabel('loss')
legend('training', 'validation')

subplot(1,3,3)
plot(x(1:10:end), acctrain(1:10:end), x(1:10:end), accval(1:10:end))
title('Accuracy')
xlabel('update steps')
ylabel('accuracy')
legend('training', 'validation')


%% Coarse search
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

result = [l;lambda;acc];
disp(['Test accuracy: ' num2str(result)]);

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
    [~, index] = max(P);
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
