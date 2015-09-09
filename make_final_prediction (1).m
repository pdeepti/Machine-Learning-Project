function prediction = make_final_prediction(model,X_test)

%% PERFORM PCA
X = [X_train; X_test];
x_bar = mean(X);
% X = bsxfun(@minus,X,x_bar); % center
[u,sqrt_eig,PCs] = fsvd(X,500); %.^2; %computes a simplified SVD on x to get the 6 biggest singular values,
%and their corresponding eigenvectors i.e the coeff/ bases for the new data.
% SVDS is better than SVD for sparse matrices 
score = X*PCs;
score_train = score(1:length(price_train),:);
score_test = score(length(price_train)+1:end,:);

%% DO K-MEANS AND LOG REGRESSION
k=3;
idx = kmeans(score,k);
price_hat = zeros;
for i=1:k
    ind = (idx==k); %logical indicator of the k-classified entries in score
    ind_train = ind(1:length(score_train)); %take only those indices which are for score_train
    ind_test = ind(length(score_train)+1:end); %take only those indices which are for score_test
    
    score_k = score(ind,:);
    score_k_train = score(ind_train,:);
    score_k_test = score(ind_test,:);
    
    price_k = price_train(ind_train);
  
    mod = glmfit(score_k_train,price_k,'normal','link','log');
    price_hat(ind_test) = glmval(mod, score_k_test,'log');
end
prediction =price_hat';
