clear;
load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat
addpath ./glmnet_matlab

X_train =[city_train word_train bigram_train];
Y_train = price_train;
X_test = [city_test word_test bigram_test];

%% PERFORM PCA ON SELECTED DATA

%divide by city
price_city = zeros;
data_y = Y_train;
X_train_rev = X_train;
X_train_rev(find(X_train_rev(:,1)),3) = 1;
X_train_rev(find(X_train_rev(:,2)),4) = 1;
X_test_rev = X_test;
X_test_rev(find(X_train_rev(:,1)),3) = 1;
X_test_rev(find(X_train_rev(:,2)),4) = 1;


for i=3:7
    %i
    ind_train_city = find(X_train_rev(:,i));
    ind_test_city = find(X_test_rev(:,i));
    Y_city = data_y(ind_train_city);
    
    X_city_train = X_train_rev(ind_train_city,8:end);
    X_city_test = X_test_rev(ind_test_city,8:end); 
    X_city = [X_city_train; X_city_test];

    [~,~,PCs]=fsvd(X_city,500);
    score_city = X_city*PCs;
    score_train_city = score_city(1:length(Y_city),:);
    score_test_city = score_city(length(Y_city)+1:end,:);
    
    gen_lin_mod = glmnet(score_train_city,Y_city);
    temp = glmnetPredict(gen_lin_mod, score_test_city);
    temp = temp(:,end);
    price_city_glm(ind_test_city) = temp;
end 

predictions = price_city_glm';
%% Write Results

dlmwrite('submit.txt',predictions,'precision','%d');