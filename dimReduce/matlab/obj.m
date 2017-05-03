function [ res ] = obj( Y, X, B, H, w, lambda1, lambda2, lambda3, lambda4,trXTX )
%OBJ Summary of this function goes here
%   Detailed explanation goes here
%[m, n] = size(X);
ntr = size(Y,1);
%loss
l = sum( max(0, 1 - Y.*(H(:,1:ntr)'*w)) );
BTB = B'*B;
HHT = H*H';
norm_approx = trXTX - 2*trace(H*X'*B) + trace(BTB*HHT); 
res = l + lambda1/2*norm(w, 'fro')^2 + lambda2/2*norm_approx + lambda3/2*norm(B,'fro')^2 + lambda4/2*norm(H,'fro')^2;
end

