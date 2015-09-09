function [ bins ] = binning( data_x, data_y, num_bins )
%binning( data_x, data_y, num_bins )
% This can either take in both x-data and y-data, OR take in only y-data.
% In both cases, the data must be ONE COLUMN of size Nx1.
% In both cases, the y-data will be divided into num_bins # of bins.
%Output is the average of each bin, size NUM_BINSx1

%If x-data is provided, the y-bins will be averaged based on the division
%of the x-data into equally sized bins. If no x-data is provided, the
%y-data will simply be divided into bins and averaged.

if nargin<3
    num_bins = 10;
end

bins = zeros;
if data_x = []
    sort_y = sort(data_y);
    [counts,centers] = hist(sort_y,num_bins);
    bins = data_y(
else
    all_data = [data_x;data_y];
    sort_x = sortrows(all_data);
    
end

