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

%initialize_additional_features;

%% Run algorithm
% % FIRST ELIMINATE FEATURES using Feat_Correlation
% top_feats = Feat_Correlation(X_train, Y_train);
% 
% numfeats = 5000;
% X_train_selected = X_train(:,top_feats(1:numfeats));
% X_test_selected = X_test(:,top_feats(1:numfeats));

%% PERFORM PCA ON SELECTED DATA
%divide by city
clearvars -except X_train X_test Y_train Y_binned
tic
price_city = zeros;
data_y = Y_binned;
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
    Y_city = data_y(ind_train_city);
    
    X_city_train = X_train_rev(ind_train_city,8:end);
    X_city_test = X_test_rev(ind_test_city,8:end);
    
%     ordered_feats = Feat_Correlation(X_city_train,Y_city);

%     X_city_train_sel = X_city_train(:,ordered_feats(1:end-100));
%     X_city_test_sel = X_city_test(:,ordered_feats(1:end-100));    
    X_city = [X_city_train; X_city_test];

    
    [~,~,PCs]=fsvd(X_city,500);
    score_city = X_city*PCs;
    score_train_city = score_city(1:length(Y_city),:);
    score_test_city = score_city(length(Y_city)+1:end,:);
    
    gen_lin_mod = glmnet(score_train_city,Y_city);
    temp = glmnetPredict(gen_lin_mod, score_test_city);
    temp = temp(:,end);
    price_city(ind_test_city) = temp;
end 
prices = price_city';
toc
%%    
X = [X_train; X_test];
%X=[X_train_selected;X_test_selected];
clearvars -except X Y_train top_feats
%x_bar = mean(X);
% X = bsxfun(@minus,X,x_bar); % center
[~,~,PCs] = fsvd(X,1500); %.^2; %computes a simplified SVD on x to get the 6 biggest singular values,
%and their corresponding eigenvectors i.e the coeff/ bases for the new data.
% SVDS is better than SVD for sparse matrices 
score = X*PCs;
score_train = score(1:length(Y_train),:);
score_test = score(length(Y_train)+1:end,:);

clearvars -except score_t* Y_train top_feats

%% Run LIBLINEAR with solve option 's 6'
% score_train_sparse = sparse(score_train);
% score_test_sparse = sparse(score_test);
% mod_PCA_libs6 = liblinear_train(Y_train, score_train_sparse, ['-s 6']);
% prices = liblinear_train(rand(size(score_test,1),1), score_test_sparse, ['-s 6']);
% 
%% DO K-MEANS AND LOG REGRESSION - kmeans wasn't great
% k=6;
% idx = kmeans(score,k);
% prices = zeros(length(score_test),1);
% k_prices= zeros(length(score_test),1);
% ind_test_all = zeros;
% for k=1:6
%     k
%     ind = (idx==k); %logical indicator of the k-classified entries in score
%     ind_train = ind(1:length(score_train)); %take only those indices which are for score_train
%     ind_test = ind(length(score_train)+1:end); %take only those indices which are for score_test
%     
%     score_k = score(ind,:);
%     score_k_train = score(ind_train,:);
%     score_k_test = score(ind_test,:);
%     
%     price_k = Y_train(ind_train);
%   
%     gen_lin_mod = glmfit(score_k_train,price_k,'normal','link','log');
%     prices(ind_test) = glmval(gen_lin_mod, score_k_test,'log');
%     %price_hat = k_prices + price_hat;
%     %ind_test_all(ind_test)=1;
% end

gen_lin_mod = glmfit(score_train,Y_train,'normal','link','log');
prices = glmval(gen_lin_mod, score_test,'log');

%% Save results to a text file for submission
dlmwrite('submit.txt',prices,'precision','%d');