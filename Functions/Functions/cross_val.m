function [ rmse ] = cross_val( data, labels, model, k_folds )
% CURRENTLY SET UP WITH LIBLINEAR, AS SCRIPT NOT AS FUNCTION
%cross_val(data,labels,'model',k_folds) takes in data, labels, your model, and the desired number of folds you want, and
%randomly subsamples your data in order to provide cross-validation error on
%your model. Error is RMSE.
%
%data: observations X features
%model: Should be a string. Takes in training and testing data as well as training labels and a lambda, outputs predictions
%folds: a scalar number, which defines how many ways your data will be split. If unspecified, data will be divided into 10 folds

%% Create Folds

data = score_train;
labels= Y_train;
k_folds = 5;

num_obsv = size(data,1);
obsv_per_fold = ceil(num_obsv/k_folds);
extras = k_folds-mod(num_obsv,k_folds);
all_ind = [1:num_obsv,randperm(size(data,1),extras)];
 n = length(all_ind);
 p = randperm(n);
rand_ind = all_ind(p);
Indices = reshape(rand_ind,obsv_per_fold,k_folds);

folds = cell(1,k_folds);
for i = 1:k_folds
    folds{i} = Indices(:,i);
end

%% Make Predictions

clearvars -except k_folds data labels folds obsv_per_fold rmse* top_feats
Label_Pred = zeros;

for i = 1:k_folds    % for each of the k_folds
    i
    trainInd = zeros;
    Testfolds = 1:k_folds;
    Testfolds(i) = 0;   % set fold "i" as the testing fold
    Testfolds = Testfolds(find(Testfolds));
    for j = 1:k_folds-1
       trainInd(1:obsv_per_fold,j) = folds{Testfolds(j)}(:); %get all training indices
    end
    trainInd = reshape(trainInd, [],1);
    
    foldtrain = data(trainInd, :);   %get corresponding training Feats
    foldtrainY = labels(trainInd);   %get labels for training Feats
    
    testInd = folds{i}(:);
    foldtest = data(testInd, :);
    foldTestY = labels(testInd);
%LIBLINEAR MODEL    
%     cv_model = liblinear_train(foldtrainY, foldtrain, ['-s 6']); %s6 gave smallest cv error. L1-regularized Logistic Regression
%     [Prediction,~,~] = liblinear_predict(rand(size(foldTestY)), foldtest, cv_model);

%GLMFIT MODEL
    gen_lin_mod = glmfit(foldtrain,foldtrainY,'normal','link','log');
    Prediction = glmval(gen_lin_mod, foldtest,'log');
    Label_Pred(testInd) = Prediction;
end
    
rmse_glm_selected=sqrt(sum((labels-Label_Pred').^2)/numel(labels))

end

