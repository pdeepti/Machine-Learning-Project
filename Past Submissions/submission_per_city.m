%% Run algorithm
% % FIRST ELIMINATE FEATURES using Feat_Correlation
% top_feats = Feat_Correlation(X_train, Y_train);
% 
% numfeats = 5000;
% X_train_selected = X_train(:,top_feats(1:numfeats));
% X_test_selected = X_test(:,top_feats(1:numfeats));

%% PERFORM PCA ON SELECTED DATA
%divide by city
clearvars -except X_train X_test Y_train
price_city = zeros;
X_train_rev = X_train;
X_train_rev(find(X_train_rev(:,1)),3) = 1;
X_train_rev(find(X_train_rev(:,2)),4) = 1;
X_test_rev = X_test;
X_test_rev(find(X_train_rev(:,1)),3) = 1;
X_test_rev(find(X_train_rev(:,2)),4) = 1;


for i=3:7
    i
    ind_train_city = find(X_train_rev(:,i));
    ind_test_city = find(X_test_rev(:,i));
    Y_city = Y_train(ind_train_city);
    
    X_city_train = X_train_rev(ind_train_city,8:end);
    X_city_test = X_test_rev(ind_test_city,8:end);
    X_city = [X_city_train; X_city_test];
    
    [~,~,PCs]=fsvd(X_city,500);
    score_city = X_city*PCs;
    score_train_city = score_city(1:length(Y_city),:);
    score_test_city = score_city(length(Y_city)+1:end,:);
    
    gen_lin_mod = glmfit(score_train_city,Y_city,'normal','link','log');
    temp = glmval(gen_lin_mod, score_test_city,'log');
%     temp(temp>20)=20;
    price_city(ind_test_city) = temp;
end  

%% Save results to a text file for submission
dlmwrite('submit.txt',prices,'precision','%d');