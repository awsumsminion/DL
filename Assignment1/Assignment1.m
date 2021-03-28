clear all; close all; clc;
addpath /Datasets/cifar-10-batches-mat;
training = 'data_batch_1.mat';
validation = 'data_batch_2.mat';
testing = 'test_batch.mat';

%% Top Level
K = 10; n = 10000; d = 3072;
mean_value = 0; dev = 0.01; 
h = 1e-6; eps = 1e-6; rng('default');

% training data
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

W = dev.*randn(K,d) + mean_value;
b = dev.*randn(K,1) + mean_value;

%% 1.7 Gradients
nb = 1; % max n = 10000
db = d; % max d = 3072
lambda = 0;
Ptrain = EvaluateClassifier(trainX, W, b);

% computing gradients analytically
[grad_b, grad_W] = ComputeGradients(trainX(1:db, nb), trainY(:, nb), Ptrain(:, nb), W(:, 1:db), lambda);

% computing gradients numerically
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX(1:db, nb), trainY(:, nb), W(:, 1:db), b, lambda, 1e-6);

% relative error
error_W = abs(norm(grad_W)-norm(ngrad_W)) ./ max(eps, norm(abs(grad_W))+norm(abs(ngrad_W)));
error_b = abs(norm(grad_b)-norm(ngrad_b)) ./ max(eps, norm(abs(grad_b))+norm(abs(ngrad_b)));

%% 1.8 Mini-batch Gradient Descent 
n_batch = 100; % the size of the mini-batches
eta = 0.001; % the learning rate
n_epochs = 40; % the number of runs through the whole training set
lambda = 1;
GDparams = [n_batch, eta, n_epochs];

% training
Wstar = W; bstar = b;
for i = 1:n_epochs
    for j=1:n/n_batch % generates mini-batches for one epoch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = trainX(:, j_start:j_end); % (dxnb)
            Ybatch = trainY(:, j_start:j_end); % (Kxnb) 
            
            Xval = valX(:, j_start:j_end); % (dxnb)
            Yval = valY(:, j_start:j_end); % (Kxnb) 
            
            [Wstar, bstar] = MiniBatchGD(Xbatch, Ybatch, GDparams, Wstar, bstar, lambda);
    end
    Jtrain(i) = ComputeCost(Xbatch, Ybatch, Wstar, bstar, lambda);
    Jval(i) = ComputeCost(Xval, Yval, Wstar, bstar, lambda);
end

Pstar1 = EvaluateClassifier(trainX, Wstar, bstar);
trainacc = ComputeAccuracy(Pstar1,trainy);

Pstar2 = EvaluateClassifier(valX, Wstar, bstar);
valacc = ComputeAccuracy(Pstar2,valy);

Pstar3 = EvaluateClassifier(testX, Wstar, bstar);
testacc = ComputeAccuracy(Pstar3,testy);

%% Results
disp('--------------------------------------');
disp(['Training accuracy: ' num2str(trainacc)]);
disp(['Training J after first epoch: ' num2str(Jtrain(1))]);
disp(['Training J after last epoch: ' num2str(Jtrain(end))]);
disp(' ');
disp(['Validation accuracy: ' num2str(valacc)]);
disp(['Validation J after first epoch: ' num2str(Jval(1))]);
disp(['Validation J after last epoch: ' num2str(Jval(end))]);
disp(' ');
disp(['Test accuracy: ' num2str(testacc)]);

% cost function plot
figure(1)
plot(1:n_epochs, Jtrain, 1:n_epochs, Jval)
title('Cost function')
xlabel('epochs')
ylabel('loss')
legend('training', 'validation')

% weight matrix images
figure(2)
for i=1:10
    im = reshape(Wstar(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:))); 
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
    subplot(1,10,i)
    imshow(s_im{i})
end

%% Functions

function [X, Y, y] = LoadBatch(fname)
% X contains image pixel data (dxn)
% Y contains the one-hot representation (Kxn)
% y contains the label for each image (nx1)
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

function P = EvaluateClassifier(X, W, b)
% X contains image pixel data (dxn)
% W (Kxd) and b (Kx1) are the parameters of the network
% col of P contains probability for each label for the image
%   corresponding to col of X (Kxn)
    fprintf('Loading EvaluateClassifier... ');
    K = size(W,1); n = size(X,2);
    for i = 1:n
        s = W*X(:,i)+b; % eq 1 (Kxd)*(dx1)+(Kx1)=(Kx1)
        P(:,i) = softmax(s); % (Kxn)
    end
    disp('Done!');
end

function acc = ComputeAccuracy(P, y)
% X (dxn), y truth labels (nx1)
% acc scalar containing the accuracy
    fprintf('Loading ComputeAccuracy... ');
    n = size(P,2);
    P_max_col = max(P);
    acc = 0;
    for i = 1:n
        p = P(:,i); % each col (Kx1)
        pmax = P_max_col(i); % max in that col
        index = 1;
        while p(index) ~= pmax
            index = index + 1;
        end
        if y(i)==index
            acc = acc + 1;
        end
    end
    acc = acc/n;
    disp('Done!');
end

function [grad_b, grad_W] = ComputeGradients(X, Y, P, W, lambda)
% X data (dxn), Y one-hot (Kxn), P probability (Kxn), W (Kxd)
% grad_W is the gradient matrix of the cost J relative to W (Kxd)
% grad_b is the gradient vector of the cost J relative to b (Kx1)
    nb = size(Y,2);
    G = -(Y-P); % (Kxn)
    L_W = (1/nb)*G*X'; % (Kxn)*(nxd) = (Kxd)
    L_b = (1/nb)*G*ones(nb,1); % (Kxn)*(nx1) = (Kx1)
    grad_W = L_W + 2*lambda*W;
    grad_b = L_b;
end


function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)
% X training images (dxn), Y labels (Kxn)
% W (Kxd) and b (Kx1) initial values, GDparams = [n_batch, eta, n_epochs]
    n_batch = GDparams(1); eta = GDparams(2); 

    P = softmax(W*X+b*ones(1,n_batch)); % (Kxnb)
    [grad_b, grad_W] = ComputeGradients(X, Y, P, W, lambda);

    Wstar = W - eta*grad_W; % (Kxd)
    bstar = b - eta*grad_b; % (Kx1)
end

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
