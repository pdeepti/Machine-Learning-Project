function [ ordered_feats ] = Feat_Correlation( data_train, labels )
%Feat_Correlation( data_train, labels ) lists the features in order of
%importance as determined by their correlation with labels

% x_bar = repmat(mean(data_train),size(data_train,1));
% centered_data = bsxfun(@minus, data_train, mean(data_train)); %could speed up both of these lines
% norm_data = bsxfun(@rdivide, centered_data, std(data_train)); %using the fast repmat he gave us
% norm_labels = (labels-mean(labels))./std(labels);

%compute linear regression
%[R, weights, intercept] = regression(norm_data, norm_labels);
rho = corr(data_train, labels);
rho_sort = sortrows([rho,[1:size(data_train,2)]']);
ordered_feats = rho_sort(:,2);
end

