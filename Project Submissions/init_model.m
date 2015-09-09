function model = init_model()

addpath libsvm

clear;
load X_train.mat
load X_test.mat
load Y_train.mat

X =X_train;
Y = Y_train;
X_t = X_test;

%% PERFORM PCA-SVM ON SELECTED DATA
X_train_rev = X;
X_test_rev = X_t;
model_svm = cell(7,1);
model_PCs = cell(7,1);
concat_Y_test = zeros;
Y_hat = zeros;

for i = 1:7
    ind_train_city = find(X_train_rev(:,i));
    ind_test_city = find(X_test_rev(:,i));
    
    X_city_train = X_train_rev(ind_train_city,8:end);
    X_city_test = X_test_rev(ind_test_city,8:end);
        
    X_city = [X_city_train; X_city_test];

    [~,~,PCs]=fsvd(X_city,250);
    score_city = X_city*PCs;
    
    Y_city{i} = Y(ind_train_city, :);
    score_train_city{i} = score_city(1:size(ind_train_city, 1),:);
    score_test_city{i} = score_city(size(ind_train_city, 1)+1:end,:);
    
     X_train = score_train_city{i};
     Y_train = Y_city{i};
     X_test = score_test_city{i};

     model_svm{i} = svmtrain(Y_train, X_train, '-s 3 -c 15');
     model_PCs{i} = PCs;
end

model.svm = model_svm;
model.PCs = model_PCs;