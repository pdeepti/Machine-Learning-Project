function [ label_prediction ] = kmeans_class( data_train, data_test,labels,num_clust )
%kmeans_class( data_train, data_test,labels,num_clust ) takes inputs of
%testing and training data, as well as the number of clusters desired. It
%outputs prediction labels.
%   Classification is done using the glmfit algorithm, and is performed on
%   each cluster. Training and Testing data must have the same number of
%   features.

data = [data_train;data_test];
idx = kmeans(data,num_clust); %divide data into clusters

% Initiate variables
label_prediction = zeros(length(data_test),1);
%k_prices= zeros(length(data_test),1);
%ind_test_all = zeros;

for k=1:num_clust
    ind = (idx==k); %logical indicator of the k-classified entries in score
    ind_train = ind(1:length(data_train)); %take only those indices which are for score_train
    ind_test = ind(length(data_train)+1:end); %take only those indices which are for score_test
    
    %data_k = data(ind,:);
    data_k_train = data(ind_train,:);
    data_k_test = data(ind_test,:);
    
    labels_k = labels(ind_train);
  
    mod = glmfit(data_k_train,labels_k,'normal','link','log');
    label_prediction(ind_test) = glmval(mod, data_k_test,'log');
    %price_hat = k_prices + price_hat;
    %ind_test_all(ind_test)=1;
end
end

