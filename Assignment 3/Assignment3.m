clear all; close all; clc;
addpath /../Datasets/cifar-10-batches-mat;
training = 'data_batch_1.mat';
validation = 'data_batch_2.mat';
testing = 'test_batch.mat';

%% Top Level
rng('default');
small_batch = 0;

% training data
if small_batch
    [trainX, trainY, trainy] = LoadBatch(training);
else
    [X1, Y1, y1] = LoadBatch('data_batch_1.mat');
    [X2, Y2, y2] = LoadBatch('data_batch_2.mat');
    [X3, Y3, y3] = LoadBatch('data_batch_3.mat');
    [X4, Y4, y4] = LoadBatch('data_batch_4.mat');
    [X5, Y5, y5] = LoadBatch('data_batch_5.mat');

    trainX = [X1 X2 X3 X4 X5(:, 1:5000)];
    trainY = [Y1 Y2 Y3 Y4 Y5(:, 1:5000)];
    trainy = [y1 ; y2 ; y3 ; y4 ; y5(1:5000)];
end

trainX = trainX./255;
mean_X = mean(trainX, 2);
std_X = std(trainX, 0, 2);

trainX = trainX - repmat(mean_X, [1, size(trainX, 2)]); 
trainX = trainX ./ repmat(std_X, [1, size(trainX, 2)]); % normalizing training data

% validation data
if small_batch
    [valX, valY, valy] = LoadBatch(validation);
else
    valX = X5(:, 5001:10000);
    valY = Y5(:, 5001:10000);
    valy = y5(5001:10000);
end

valX = valX./255;
valX = valX - repmat(mean_X, [1, size(valX, 2)]); 
valX = valX ./ repmat(std_X, [1, size(valX, 2)]); % normalizing validation data

% test data
[testX, testY, testy] = LoadBatch(testing);
testX = testX./255;
testX = testX - repmat(mean_X, [1, size(testX, 2)]); 
testX = testX ./ repmat(std_X, [1, size(testX, 2)]); % normalizing validation data


%% Gradients
%{
K = 10; n = 10000; d = 3072; 
m = [50 50 50]; % m = [m1, m2]
k = size(m,2)+1;
use_bn = 1;
lambda = 0.01;
nb = 1; % max n = 10000
db = 10; % max d = 3072
 NetParams = ParamsInit(trainX(1:db, nb), trainY(:, nb), m, use_bn); %NetParams = ParamsInit(trainX, trainY, m, use_bn);

% computing gradients analytically
Grads = ComputeGradients(trainX(1:db, nb), trainY(:, nb), NetParams, lambda);

% computing gradients numerically slow
GradsNum = ComputeGradsNumSlow(trainX(1:db, nb), trainY(:, nb), NetParams, lambda, 1e-5);

%relative error
for i = 1:k
    error_W{i} = abs(norm(Grads.W{i})-norm(GradsNum.W{i})) ./ max(eps, norm(abs(Grads.W{i}))+norm(abs(GradsNum.W{i})));
    error_b{i} = abs(norm(Grads.b{i})-norm(GradsNum.b{i})) ./ max(eps, norm(abs(Grads.b{i}))+norm(abs(GradsNum.b{i})));
    if (use_bn) && (i<k)
        error_ga{i} = abs(norm(Grads.gammas{i})-norm(GradsNum.gammas{i})) ./ max(eps, norm(abs(Grads.gammas{i}))+norm(abs(GradsNum.gammas{i})));
        error_be{i} = abs(norm(Grads.betas{i})-norm(GradsNum.betas{i})) ./ max(eps, norm(abs(Grads.betas{i}))+norm(abs(GradsNum.betas{i})));
    end
end

fprintf('Error W:'); disp(error_W)
fprintf('Error b:'); disp(error_b)
if use_bn
    fprintf('Error gamma:'); disp(error_ga)
    fprintf('Error beta:'); disp(error_be)
end

%}
%% Cyclical learning rates

K = size(trainY,1); % 10
d = size(trainX, 1); % 3072
n = size(trainX, 2); % 45000
nval = size(valX, 2); % 5000

m = [50, 50]; % hidden layers
k = size(m,2)+1;
use_bn = 1;
NetParams = ParamsInit(trainX, trainY, m, use_bn); % NetParams.W = {W1, ..., Wk}, NetParams.b = {b1, ..., bk}

eta_min = 1e-5;
eta_max = 1e-1;
n_batch = 100; 
num_cycles = 2;
%n_s = 2*floor(n/n_batch);
%n_s = 5 * n/n_batch;
n_s = 2 * n/n_batch;
n_epochs = (num_cycles*2*n_s*n_batch)/n;

lambda = 0.005;
%lambda = 0.004260370538649; % best lambda
l = 0;
t_min = 2*l*n_s;
t_mid = (2*l+1)*n_s;
t_max = 2*(l+1)*n_s;
t = 1;

Wstar = NetParams.W;
bstar = NetParams.b;

q = 50;
a = 0.9;

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

        NetParams = MiniBatchGD(XbatchT, YbatchT, eta_t, NetParams, lambda, a);
        
        if NetParams.use_bn
            mu_av = NetParams.mu_av{k-1};
            v_av = NetParams.v_av{k-1};
            
            if mod(t,q) == 1
                [c, r] = ComputeCost(XbatchT, YbatchT, NetParams, lambda, mu_av, v_av);
                Ptrain = EvaluateClassifier(XbatchT, NetParams, mu_av, v_av);
                acctrain(t) = ComputeAccuracy(Ptrain,ybatchT); Jtrain(t) = c; losstrain(t) = r;
            end

            %validation
            XbatchV = valX; % (dxnb)
            YbatchV = valY; % (Kxnb) 
            ybatchV = valy;

            if mod(t,q) == 1
                [c, r] = ComputeCost(XbatchV, YbatchV, NetParams, lambda, mu_av, v_av);
                Pval = EvaluateClassifier(XbatchV, NetParams, mu_av, v_av);
                accval(t) = ComputeAccuracy(Pval,ybatchV); Jval(t) = c; lossval(t) = r;
            end
        else
            if mod(t,q) == 1
                [c, r] = ComputeCost(XbatchT, YbatchT, NetParams, lambda);
                Ptrain = EvaluateClassifier(XbatchT, NetParams);
                acctrain(t) = ComputeAccuracy(Ptrain,ybatchT); Jtrain(t) = c; losstrain(t) = r;
            end

            %validation
            XbatchV = valX; % (dxnb)
            YbatchV = valY; % (Kxnb) 
            ybatchV = valy;

            if mod(t,q) == 1
                [c, r] = ComputeCost(XbatchV, YbatchV, NetParams, lambda);
                Pval = EvaluateClassifier(XbatchV, NetParams);
                accval(t) = ComputeAccuracy(Pval,ybatchV); Jval(t) = c; lossval(t) = r;
            end
        end
        
        t = t + 1;
        if t > t_max
            l = l + 1;
            t_min = 2*l*n_s;
            t_mid = (2*l+1)*n_s;
            t_max = 2*(l+1)*n_s;
        end
            
    end
    
    ind = randperm(n);
    trainX = trainX(:, ind);
    trainY = trainY(:, ind);
    trainy = trainy(ind, :);
end
x = 1:t-1;

Pstar = EvaluateClassifier(testX, NetParams);
testacc = ComputeAccuracy(Pstar,testy);

disp(['Test accuracy: ' num2str(testacc)]);

figure(1)
subplot(1,3,1)
plot(x(1:q:end), Jtrain(1:q:end), x(1:q:end), Jval(1:q:end))
title('Cost function')
xlabel('update steps')
ylabel('cost')
legend('training', 'validation')

subplot(1,3,2)
plot(x(1:q:end), losstrain(1:q:end), x(1:q:end), lossval(1:q:end))
title('Loss function')
xlabel('update steps')
ylabel('loss')
legend('training', 'validation')

subplot(1,3,3)
plot(x(1:q:end), acctrain(1:q:end), x(1:q:end), accval(1:q:end))
title('Accuracy')
xlabel('update steps')
ylabel('accuracy')
legend('training', 'validation')


%% Coarse search
%{
K = size(trainY,1); % 10
d = size(trainX, 1); % 3072
m = [50 50];

ntrain = size(trainX, 2); % 45000
nval = size(valX, 2); % 5000

eta_min = 1e-5;
eta_max = 1e-1;
n_batch = 100; 
num_cycles = 2;

n_s = 5 * ntrain/n_batch;
n_epochs = (num_cycles*2*n_s*n_batch)/ntrain;
use_bn = 1;

lmin = -3; lmax = -2;
for i = 1:8
    l(i) = lmin + (lmax - lmin)*rand(1, 1); 
    lambda(i) = 10^l(i);
end

GDparams = [n_batch, eta_min, eta_max, n_epochs, n_s];

for i = 1:length(lambda)
    NetParams = ParamsInit(trainX, trainY, m, use_bn); % {W1, W2, b1, b2}
    acc(i) = FindLambda(trainX, trainY, valX, valy, NetParams, GDparams, lambda(i));
end

result = [l;lambda;acc]

%}
%% Functions

function [X, Y, y] = LoadBatch(fname) 
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

function NetParams = ParamsInit(X, Y, m, use_bn)
    K = size(Y,1); d = size(X,1);
    k = size(m,2)+1; count = k;
    m = [K m d];
    sig = 1e-4; % for normal distribution
    NetParams.use_bn = use_bn; % 0 = no BN, 1 = BN
    for i = 1:k
        dev = 1/sqrt(m(i+1));
        %NetParams.W{count} = 
        x = dev.*randn(m(i),m(i+1));
        % x = randn(m(i),m(i+1));
        NetParams.W{count} = normcdf(x, 0, sig);
        NetParams.b{count} = zeros(m(i),1);
        if use_bn
            NetParams.gammas{count} = ones(m(i),1);
            NetParams.betas{count} = zeros(m(i),1);
        end
        count = count - 1;
    end
end

function P = EvaluateClassifier(X, NetParams, varargin)
    fprintf('Loading EvaluateClassifier... ');
    W = NetParams.W; b = NetParams.b;
    k = numel(W); nb = size(X,2);
    H{1} = X;
    
    for i=1:k-1
        if NetParams.use_bn
            ga = NetParams.gammas{i}; 
            be = NetParams.betas{i};
            
            s = W{i}*H{i}+b{i};
            
            if nargin == 4
                mu = varargin{1}(i);
                v = varargin{2}(i);
            else
                [mu, v] = meanvar(s);
            end
            
            s_hat = BatchNormalize(s, mu, v);
            s_tilde = ga.*s_hat+be;
            H{i+1} = max(0, s_tilde);
        else
            s = W{i}*H{i}+b{i}*ones(1,nb);
            H{i+1} = max(s,0);
        end
    end
    P = softmax(W{k}*H{k}+b{k}*ones(1,nb));
    disp('Done!');
end

function [mu, v] = meanvar(s)
    [~,n] = size(s);
    mu = mean(s,2);
    v = var(s, 0, 2);
    v = v * (n - 1) / n;
end

function s_hat = BatchNormalize(s, mu, v)
    s_hat = (diag((v+eps).^-0.5))*(s-mu); % s_hat = ((diag(v+eps))^(-0.5))*(s-mu);
end

function acc = ComputeAccuracy(P, y)
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

function Grads = ComputeGradients(X, Y, NetParams, lambda, varargin)
    W = NetParams.W; b = NetParams.b;
    k = numel(W); nb = size(Y,2); 
    H{1} = X;
    
    if NetParams.use_bn
        gammas = NetParams.gammas;
        betas = NetParams.betas;
    end
    
    if nargin == 6
        Mu = varargin{1};
        V = varargin{2};
    end

    % forward pass
    for i=1:k-1
        if NetParams.use_bn
            S{i} = W{i}*H{i}+b{i};
            
            if nargin ~= 6
                [Mu{i}, V{i}] = meanvar(S{i});
            end
            
            S_hat{i} = BatchNormalize(S{i}, Mu{i}, V{i}); % *ones(1,nb)
            s_tilde = gammas{i}.*S_hat{i}+betas{i};
            H{i+1} = max(0, s_tilde);
            
        else
            S{i} = W{i}*H{i}+b{i}*ones(1,nb);
            H{i+1} = max(S{i},0);
        end
    end
    S{k} = W{k}*H{k}+b{k};
    P = softmax(S{k});
    
    % backward pass
    G = -(Y-P);
       
    if NetParams.use_bn
        Grads.mu = Mu;
        Grads.v = V;
        
        grad_W{k} = (1/nb)*G*H{k}'+2*lambda*W{k};
        grad_b{k} = (1/nb)*G*ones(nb,1);
        G = W{k}'*G;
        G = G.*sign(H{k});
        l = k-1;
        
        for i = 1:k-1
            grad_gammas{l} = (1/nb)*(G.*S_hat{l})*ones(nb,1);
            grad_betas{l} = (1/nb)*(G*ones(nb,1));

            G = G.*(gammas{l}*ones(1,nb));
            G = BatchNormBackPass(G, S{l}, Mu{l}, V{l});
            
            grad_W{l} = (1/nb)*G*H{l}'+2*lambda*W{l};
            grad_b{l} = (1/nb)*G*ones(nb,1);

            if l > 1
                G = W{l}'*G;
                G = G.*sign(H{l});
            end

            l = l - 1;
        end
        Grads.gammas = grad_gammas;
        Grads.betas = grad_betas;
    else
        count = k;
        for i=2:k
                grad_W{count} = (1/nb)*G*H{count}'+2*lambda*W{count};
                grad_b{count} = (1/nb)*G*ones(nb,1);
                G = W{count}'*G;
                G = G.*sign(H{count});
                count = count - 1;
        end
        grad_W{1} = (1/nb)*G*X'+2*lambda*W{1};
        grad_b{1} = (1/nb)*G*ones(nb,1);   
    end
    
    Grads.W = grad_W;
    Grads.b = grad_b;
end

function Gb = BatchNormBackPass(G, S, mu, v) % output G = (m x 1)
    n = size(S,2); 
    
    sigma1 = ((v+eps).^-0.5)';
    sigma2 = ((v+eps).^-1.5)';
    
    G1 = G.*(sigma1'*ones(1,n));
    G2 = G.*(sigma2'*ones(1,n));
    
    D = S - mu*ones(1,n);
    c = (G2.*D)*ones(n,1);
    
    Gb = G1 - (1/n)*(G1*ones(n,1)*ones(1,n)) - (1/n)*D.*(c*ones(1,n));
end

function NetParams = MiniBatchGD(X, Y, eta, NetParams, lambda, a)
    Grads = ComputeGradients(X, Y, NetParams, lambda);
    
    grad_W = Grads.W; grad_b = Grads.b; 
    W = NetParams.W; b = NetParams.b;
    k = numel(W);
    
    if NetParams.use_bn
        if numel(fieldnames(NetParams)) == 7
            mu_av = NetParams.mu_av;
            v_av = NetParams.v_av;
            mu = Grads.mu;
            v = Grads.v;
            
            for l=1:k-1
                mu_av{l} = a*mu_av{l} + (1-a)*mu{l};
                v_av{l} = a*v_av{l} + (1-a)*v{l};
            end
        else
            mu_av = Grads.mu;
            v_av = Grads.v;
            
            for l=1:k-1
                mu_av{l} = a*mu_av{l} + (1-a)*mu_av{l};
                v_av{l} = a*v_av{l} + (1-a)*v_av{l};
            end
        end

        NetParams.mu_av = mu_av;
        NetParams.v_av = v_av;
    end
    
    for i = 1:k
        Wstar{i} = W{i} - eta*grad_W{i}; % (Kxd)
        bstar{i} = b{i} - eta*grad_b{i}; % (Kx1)
    end
    NetParams.W = Wstar;
    NetParams.b = bstar;
end

function val_acc = FindLambda(trainX, trainY, valX, valy, NetParams, GDparams, lambda)
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
    
    a = 0.9;
    
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

            NetParams = MiniBatchGD(XbatchT, YbatchT, eta_t, NetParams, lambda, a);

            t = t + 1;
            if t > t_max
                l = l + 1;
                t_min = 2*l*n_s;
                t_mid = (2*l+1)*n_s;
                t_max = 2*(l+1)*n_s;
            end

        end

        ind = randperm(n);
        trainX = trainX(:, ind);
        trainY = trainY(:, ind);
    end

    Pval = EvaluateClassifier(valX, NetParams);
    val_acc = ComputeAccuracy(Pval,valy);
end

function [c, r] = ComputeCost(X, Y, NetParams, lambda, varargin)
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
