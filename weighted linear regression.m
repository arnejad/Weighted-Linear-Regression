% This code has been developed by Ali Rahmani Nejad at IASBS for educational puporses

clear all
%% INITIALIZATIONS
fid = fopen('features.txt');
data(:,1) = fscanf(fid, '%f');
fid = fopen('labels.txt');
data(:,2) = fscanf(fid, '%f');  % column one coresponds to ...
                                % X (feature) and 2nd column Y (lables)

train = data(1:80,:);
test = data(81:100,:);

clear fid data;

PARAM = 10;

%% SIMPLE LINEAR REGRESSION

% No Train Phase exists !!!

feats_train = train(:,1);
labels_train = train(:, 2);

% Test phase

feats_test = test(:,1);
test_sample = feats_test(13);
test_label = test(13,2);
% test_label = test(
% building diagonal weights matrix
Weights = diag(exp(-((((test_sample-feats_train).^2)/(2 * (PARAM)^2)))));
feats_train = [ones(length(train), 1) train(:,1)];

%phi = [ones(size(feats_train,1),1) feats_train feats_train.^2 feats_train.^3 feats_train.^4 feats_train.^5];
% W = inv(phi'*Weights*phi) * phi' * Weights * labels_train;
THETA = inv(feats_train'*Weights*feats_train) * feats_train' * Weights * labels_train;

% phi_test = [ones(size(feats_test,1),1) feats_test feats_test.^2 feats_test.^3 feats_test.^4 feats_test.^5];
pred = test_sample * THETA(2) + THETA(1);

%% VISUALIZATION

figure;
scatter(feats_train(:,2), labels_train, 'blue','filled')
hold on;
% scatter(feats_test, pred, 'red','filled')
scatter(test_sample, pred , 'red','filled')
hold on;
scatter(test_sample, test_label, 'green', 'filled')
hold on;
fplot(poly2sym(flip(THETA)), 'red') % draw the function
title("simple linear regression (PARAM: " + PARAM + ")")
ylabel('label') 
xlabel('feature') 
legend('train set','prediction', 'correct label', 'prediction function')
