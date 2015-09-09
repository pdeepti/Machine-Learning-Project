% clear;
% load ../data/city_train.mat
% load ../data/city_test.mat
% load ../data/word_train.mat
% load ../data/word_test.mat
% load ../data/bigram_train.mat
% load ../data/bigram_test.mat
% load ../data/price_train.mat

% X_train =[city_train word_train bigram_train];
% Y_train = price_train;
% X_test = [city_test word_test bigram_test];
load X_test.mat
load X_train.mat
load Y_train.mat
%initialize_additional_features;

%% Run algorithm
% PERFORM PCA ON SELECTED DATA
X = [X_train; X_test];
clearvars -except X Y_train top_feats
[~,~,PCs] = fsvd(X,2000); %.^2; %computes a simplified SVD on x to get the 6 biggest singular values,
%and their corresponding eigenvectors i.e the coeff/ bases for the new data.
% SVDS is better than SVD for sparse matrices 
display('PC done')
score = X*PCs;
score_train = score(1:length(Y_train),:);
score_test = score(length(Y_train)+1:end,:);

clearvars -except score_t* Y_train top_feats

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

%GLMFIT MODEL
    gen_lin_mod = glmfit(foldtrain,foldtrainY,'normal','link','log');
    Prediction = glmval(gen_lin_mod, foldtest,'log');
    Label_Pred(testInd) = Prediction;
end
    
rmse=sqrt(sum((labels-Label_Pred').^2)/numel(labels))

save('rmse.mat','rmse')