function [B,S] = SSNMF(X, Y, k, lambda, tridx, teidx, tol, maxiter)
% X \in R^{m x n}, Y \in R^{2 x ntr}, A \in R^{m x k}, S \in R^{k x n}
[m, n] = size(X);
r = size(Y,1);
A = abs(randn(m, k));
S = abs(randn(k, n));
B = randn(r, k);
L = zeros(r, n);
ntr = size(tridx, 2);
nte = size(teidx, 2);
Y = [Y, zeros(r, nte)];
L(:,tridx) = 1;
Xnorm = norm(X, 'fro')^2;
Ynorm = norm(Y, 'fro')^2;
preobj = inf;
for i = 1:maxiter,
	fprintf('# iter %d\n', i);
	Str = S(:,tridx);
	Ste = S(:,teidx);
	SST = S*S';
	A = A.*( (X*S') ./ (A*SST) );
	%B = B.*( (Y*S(:,tridx)') ./ (L.* (B*S) *S' ) );
	B = B .* ( ((L.*Y)*S') ./ (L.*(B*S) * S') );
	S = S .* ( (A'*X + lambda * B'*(L.*Y)) ./ ( A'*A*S + lambda*B'*(L.*(B*S)) ) );
	%BTY = B'*Y;
	%BTY = [BTY, zeros(k, nte)];
	%S = S.*( (A'*X + lambda*BTY) ./ ((A'*A)*S) + lambda*B'*(L.*(B*S)) );
	Ytr = Y(:, tridx);
	obj = Xnorm + trace((A'*A)*(S*S')) - 2*trace(S*X'*A) + lambda*( Ynorm + trace((B'*B)*(S(:,tridx)*S(:,tridx)') ) - 2*trace(S(:,tridx)*Ytr'*B) );	
	obj
	err = obj - norm(X-A*S,'fro')^2 - lambda*norm(L.*(Y-B*S),'fro')^2
	if( obj > preobj ),
		error('increase in objective value');
	end	
	if( (preobj - obj)/obj <= tol ),
		break;
	end
	preobj = obj;
end

end
