function [ accuracy ] = pred( Y, H, w )
%PRED Summary of this function goes here
%   Detailed explanation goes here
if( size(Y,1) ~= size(H,2) ),
   error('dimension not match in prediction.'); 
end
n = size(Y, 1);
idx = H'*w <= 0;
Ypred = ones(n,1);
Ypred(idx) = -1;
accuracy = sum(Y == Ypred)/n;
end

