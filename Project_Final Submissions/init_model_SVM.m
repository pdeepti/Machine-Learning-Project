function svm_model = init_model_SVM()

    clear;
    load ../data/city_train.mat
    load ../data/city_test.mat
    load ../data/word_train.mat
    load ../data/word_test.mat
    load ../data/bigram_train.mat
    load ../data/bigram_test.mat
    load ../data/price_train.mat
    addpath ./libsvm
    
    X_train_all_cities =[city_train word_train bigram_train];
    Y_train_all_cities = price_train;
    X_test_all_cities = [city_test word_test bigram_test];
    
    for i = 1:7
    
        ind_train_city = find(X_train_all_cities(:,i));
        ind_test_city = find(X_test_all_cities(:,i));
 
        X_city_train = X_train_all_cities(ind_train_city,8:end);
        X_city_test = X_test_all_cities(ind_test_city,8:end);
        X_city = [X_city_train; X_city_test];
    
%         PCs{i} = init_model_PCA(X_city_train, X_city_test, 1:7, 250);
        [~,~,PCs{i}]=fsvd(X_city,250);
        score_city = X_city*PCs{i};
        Y_city{i} = Y_train_all_cities(ind_train_city, :);
        score_train_city{i} = score_city(1:size(ind_train_city, 1),:);
        score_test_city{i} = score_city(size(ind_train_city, 1)+1:end,:);
        
    end
        
    concat_Y_test = [];
    Y_hat = [];

    for i=1:7
    
        X_train = score_train_city{i};
        Y_train = Y_city{i};
        X_test = score_test_city{i};
        Y_test = zeros(size(X_test,1),1);

        svm_mod{i} = svmtrain(Y_train, X_train, '-s 3 -c 15');
    end
  svm_model = [PCs svm_mod];  
    