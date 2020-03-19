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

%% SIMPLE LINEAR REGRESSION

% Train phase
feats_train = train(:,1);
labels_train = train(:, 2);
phi = [ones(size(feats_train,1),1) feats_train feats_train.^2 feats_train.^3 feats_train.^4 feats_train.^5];
W = inv(phi'*phi) * phi' * labels_train;

% Test phase
feats_test = test(:,1);
phi_test = [ones(size(feats_test,1),1) feats_test feats_test.^2 feats_test.^3 feats_test.^4 feats_test.^5];
pred = phi_test * W;

%% VISUALIZATION

figure;
scatter(feats_train, labels_train, 'blue','filled')
hold on;
scatter(feats_test, pred, 'red','filled')
hold on;
fplot(poly2sym(flip(W)), 'red') % draw the function
title("simple linear regression result")
ylabel('label') 
xlabel('feature') 