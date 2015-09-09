function [ binned_data, bin_avg, binIdx ] = bin_y(data_y, nbins )
%binning( data_x, data_y, num_bins )
% This takes in only y-data, as well as number of bins.
% The data must be ONE COLUMN of size Nx1.
% The y-data will be divided into num_bins # of bins.
% Output is the average of each bin, size NUM_BINSx1

if nargin<2
    nbins = 10;
end

binEdges = linspace(min(data_y),max(data_y),nbins+1);
%lowLim = binEdges(1:end-1);
%upLim = binEdges(2:end);
[~,binIdx] = histc(data_y, [binEdges(1:end-1) Inf]);
bin_avg = accumarray(binIdx, data_y, [], @mean);
binned_data = data_y;
for i=1:nbins
    binned_data(binIdx==i) = bin_avg(i);
end
end

