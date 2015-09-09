clear;
load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat
%addpath ./glmnet_matlab
%addpath ./liblinear
addpath ./libsvm
    
    concat_Y_test = [];
    Y_hat = [];

    for i=1:7
        
        ind_test_city = find(X_test(:,i));
        X_city_test = X_test(ind_test_city,8:end);
        X_city = [X_city_train; X_city_test]
        Y_test = zeros(size(X_test,1),1);

        [labels_svm acc] = svmpredict(Y_test, X_test, model);
        concat_Y_test = vertcat(concat_Y_test, Y_test);
        Y_hat = vertcat(Y_hat, labels_svm);   
    end 