function prediction = make_final_prediction(model,X_test,temp)

addpath libsvm
city = find(X_test(1:7)==1);
score_test = X_test(:,8:end)*model.PCs{city};
Y_rand = zeros(size(score_test,1),1);
[prediction, ~] = svmpredict(Y_rand, score_test, model.svm{city});
end