function [X_test X_train Y_test Y_train] = make_partitions(X,Y,ratio)


%dummy = [1 1;2 2;3 3;4 4;5 5;6 6;7 7;8 8;9 9;10 10];
total_size = size(X,1);
train_size = floor(ratio*total_size);
r=randperm(total_size,train_size);
total=1:size(X, 1);
train_indices = total(r);
test_indices = setdiff(total',r', 'rows');
X_train = X(train_indices,:);
X_test = X(test_indices,:);
Y_train = Y(train_indices,:);
Y_test = Y(test_indices,:);