function [ res ] = obj( Y, X, tridx, teidx, B, H, w, lambda1, lambda2, lambda3,trXTX, loss, preobj )
%OBJ Summary of this function goes here
%   Detailed explanation goes here
%[m, n] = size(X);
lambda4 = lambda3;
ntr = size(Y,1);
%loss
if strcmp(loss, 'l1'),
l = sum( max(0, 1 - Y.*(H(:,tridx)'*w)) );
else
l = sum( max(0, 1 - Y.*(H(:,tridx)'*w)).^2 );
end
BTB = B'*B;
HHT = H*H';
norm_approx = trXTX - 2*trace(H*X'*B) + trace(BTB*HHT); 
res = l + lambda1/2*norm(w, 'fro')^2 + lambda2/2*norm_approx + lambda3/2*norm(B,'fro')^2 + lambda4/2*norm(H,'fro')^2;
if strcmp(loss, 'l2') && res > preobj,
	error('increase in objective value');
end

end

