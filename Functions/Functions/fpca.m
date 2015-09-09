function [ score, princ_comps ] = fpca( X_train, X_test )
%fpca(X_train, X_test) outputs [score, principle components]
%   score: #observation x #components -sized matrix, where the observations are
%   the TOTAL NUMBER of observations for BOTH the training and testing
%   data, with the training data coming first and the testing data coming
%   second. "score" is this data projected into each PC, respectively

X = [X_train; X_test];
% x_bar = mean(X);
% X = bsxfun(@minus,X,x_bar); % center
[~,~,princ_comps] = fsvd(X,500); %.^2; %computes a simplified SVD on x to get the 6 biggest singular values,
%and their corresponding eigenvectors i.e the coeff/ bases for the new data.
% SVDS is better than SVD for sparse matrices 
score = X*princ_comps;

end

