function [ rmse ] = compute_rmse( predicted, price_test )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
actual = price_test;
rmse = sqrt(sum((actual-predicted).^2)/numel(actual));

end

