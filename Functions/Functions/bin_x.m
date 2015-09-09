function [ bins ] = bin_x(data_x, data_y, nbins )
%binning( data_x, data_y, num_bins )
% This takes in both x-data and y-data, as well as nbins. If nbins is
% unspecified, 10 bins are used.
% In both cases, the data must be ONE COLUMN of size Nx1.
% Output is the average of each bin, size NUM_BINSx1

% The y-bins are averaged based on the division of the X-DATA into equally
% sized bins. In other works, each bin has the average y-label for a bin of
% X-data.

if nargin<3
    nbins = 10;
end

binEdges = linspace(min(data_x),max(data_x),nbins+1);
lowLim = binEdges(1:end-1);
upLim = binEdges(2:end);
[~,binIdx] = histc(data_x, [binEdges(1:end-1) Inf]);
bins = accumarray(binIdx, data_y, [], @mean);
    
end

