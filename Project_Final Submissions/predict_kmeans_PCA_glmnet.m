clear;
load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat

X_train =[city_train word_train bigram_train];
Y_train = price_train;
X_test = [city_test word_test bigram_test];

%% PCA, glmnet and k-means

%divide by city
multi_svm_pred = zeros;
X_train_rev = X_train;
X_train_rev(find(X_train_rev(:,1)),3) = 1;
X_train_rev(find(X_train_rev(:,2)),4) = 1;
X_test_rev = X_test;
X_test_rev(find(X_train_rev(:,1)),3) = 1;
X_test_rev(find(X_train_rev(:,2)),4) = 1;

data_y = Y_train;

for i=3:7
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

    %Do k-means within each city to partition data    
    k=6;
    idx = kmeans(score_city,k);
    k_prices= zeros(length(score_test_city),1); %all prices for the city
    
    for k=1:6
        ind = (idx==k); %'ind' is a logical indicator, maps from the city to the particular cluster
        k_ind_train = ind(1:length(score_train_city)); %take only those indices which are for score_train, in the cluster
        k_ind_test = ind(length(score_train_city)+1:end); %take only those indices which are for score_test, in the cluster

        score_k = score_city(ind,:);
        score_k_train = score_city(k_ind_train,:);
        score_k_test = score_city(k_ind_test,:);

        label_k = Y_city(k_ind_train);
        
        gen_lin_mod = glmnet(score_k_train,label_k);
        temp = glmnetPredict(gen_lin_mod, score_k_test);
        temp = temp(:,end);
        k_prices(k_ind_test) = temp;
    end
    multi_svm_pred(ind_test_city) = k_prices;

end
prices = multi_svm_pred';