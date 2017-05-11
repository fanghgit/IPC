function [ B, H, w, acc_tr_list, acc_te_list ] = drsvm( Y, X, tridx, teidx, k, lambda1, lambda2, lambda3, maxiter, tol, loss, nonneg, Ytest )
%DRSVM Summary of this function goes here
%   min sum l(y_i, h_i, w) + lambda1/2 \|w\|_2^2 + lambda2/2 \|X - BH\|_F^2 +
%   lambda3 \|B\|_F^2 + lambda3 \|H\|_F^2
if ~strcmp(loss, 'l1') && ~strcmp(loss, 'l2'),
    error('invalid loss function');
end
acc_tr_list = [];
acc_te_list = [];
[m, n] = size(X);
B = randn(m, k);
H = randn(k, n);
w = randn(k, 1);
ntr = size(Y, 1);
%XTX = X'*X;
trXTX = norm(X, 'fro')^2;
lambda4 = lambda3;
pre_obj = inf;
obj_value = obj(Y, X, tridx, teidx, B, H, w, lambda1, lambda2, lambda3, trXTX, loss, pre_obj)
pre_obj = obj_value;
unknown_idx = [1:n];
unknown_idx(tridx) = [];
%sgd_t = 1;
for iter = 1:maxiter
    fprintf('%d iteration.\n', iter);
    %Update B    
    %tic;
    HHT = H*H';
    XHT = X*H';
    r = lambda3/lambda2;
    for i = 1:k,
	hjnorm = HHT(i,i);
        B(:,i) = hjnorm/(hjnorm+r)*B(:,i) + ( XHT(:,i) - B*HHT(:,i) ) / (hjnorm + r);
	if nonneg,   %non-negative setting
	    B(:,i) = max(0, B(:,i));
	end
    end

    %HHT = H*H';
    %HXT = H*X';
    %BT = (HHT + lambda3/lambda2*eye(k)) \ HXT;
    %B = BT';
    %toc
    %disp('Update B')
    obj_value = obj(Y, X, tridx, teidx, B, H, w, lambda1, lambda2, lambda3, trXTX, loss, pre_obj)
    %pre_obj = obj_value;
    
    %Update Htrain
    %tic;
    BTB = B'*B;
    disp('BTB');
    
    XtrTB = X(:,tridx)'*B;
    XteTB = X(:,unknown_idx)'*B;
    %toc
    for i = 1:k,
   %    disp('frist');
   %    tic
       %bjnorm = norm(B(:,i), 'fro')^2;
       idx = [1:k];
       idx(i) = [];
       bjnorm = BTB(i,i);
       %tic;
       r = lambda4/lambda2;
       sol1 = H(i,tridx)'*bjnorm/(bjnorm + r) + ( XtrTB(:,i) - H(:,tridx)'*(BTB(:,i))  ) / (bjnorm + r);
       if strcmp(loss, 'l1'),    % l1 
           sol2 =  H(i,tridx)'*bjnorm/(bjnorm + r) + ( XtrTB(:,i) - H(:,tridx)'*(BTB(:,i))  ) / (bjnorm + r)  + w(i,1)*Y/(lambda2*(bjnorm + r));
       else,			 % l2
           q = 2*w(i,1)^2/lambda2;
           wTHj = H(idx,tridx)'*w(idx,1);
	   sol2 = H(i,tridx)'*bjnorm/(bjnorm + r + q) + ( XtrTB(:,i) - H(:,tridx)'*(BTB(:,i))  ) / (bjnorm + r + q) + 2/lambda2*w(i,1)*( Y.*(1 - Y.*wTHj) )/(bjnorm + r + q);
       end
	%disp('calculating sol');
       
       H(i,unknown_idx) = H(i,unknown_idx)*bjnorm/(bjnorm + r) + (XteTB(:,i)' - BTB(:,i)'*H(:,unknown_idx) ) / (bjnorm + r); 
       if nonneg,   %non-negative setting
	   H(i, unknown_idx) = max(0, H(i, unknown_idx));
       end       

       %idx = [1:k];
       %idx(i) = [];
       thresold = (1 - Y.*(H(idx,tridx)' * w(idx,:) )) ./ (Y*w(i, 1));
       %toc
       %    toc
    %   disp('second')
    %   tic
       sol = sol1;
       pos_deno = Y*w(i,1) > 0;
       neg_deno = Y*w(i,1) < 0;
       zero_deno = Y*w(i,1) == 0;
       if sum(zero_deno) > 0,
          error('error: zero denominator') 
       end
       %size(sol1)
       %ii = sol1 >= thresold;
       pos1 = ( pos_deno & (sol1 >= thresold) ) | (neg_deno & (sol1 <= thresold));
       pos2 = ( pos_deno & (sol2 < thresold) ) | (neg_deno & (sol2 > thresold));
       if( sum(pos1 & pos2) ~= 0 ),
          sum(pos1 & pos2)
          error('error: pos1 & pos2 in not empty') 
       end
       pos3 = ~(pos1) & ~(pos2);
       sol(pos1) = sol1(pos1);
       sol(pos2) = sol2(pos2);
       sol(pos3) = thresold(pos3);
       
      % toc
       H(i,tridx) = sol;
       if nonneg,
	   H(i,tridx) = max(0, H(i,tridx));
       end
       %fprintf('i = %d \n', i);       
    end
    
    %disp('Update Htr');
    %toc
    %obj(Y, X, tridx, teidx, B, H, w, lambda1, lambda2, lambda3, lambda4, trXTX)
    obj_value = obj(Y, X, tridx, teidx, B, H, w, lambda1, lambda2, lambda3, trXTX, loss, pre_obj)
    %pre_obj = obj_value;

    %Update w  Pegasos algorithm
    %tic
    %size(Y)
    %size(H(:,1:ntr)')
    l = sum( max(0, 1 - Y.*(H(:,tridx)'*w)) ) + lambda1/2*w'*w
    if strcmp(loss, 'l1'),
	par = ['-c ' num2str(1/lambda1) ' -s 3 e 0.0000001' ];
    else,
	par = ['-c ' num2str(1/lambda1) ' -s 2' ];
    end
    %keyboard
    model = train(Y, sparse(H(:,tridx)'), par, w);
    
    w = model.w';
    %size(w)
    % T = 5*ntr;
    % for t = 1:T,
    %     ti = randi(ntr);
    %     eta = 1/(t*ntr*lambda1);
    %     if Y(ti,1)*(w'*H(:,ti)) < 1,
    %         w = (1 - eta*lambda1) * w  + eta*Y(ti,1)*H(:,ti); 
    %     else
    %         w = (1 - eta*lambda1) * w;
    %     end
%         sgd_t = sgd_t + 1;
    % end
  
    
%     eta = 1e-1;
%     for i = 1:n,
%         subgrad = ( Y(i)*w'*H(:,i) <= 1 )*(-Y(i)*H(:,i)) + lambda1*w;
%         w = w - eta*subgrad;
%     end
    disp('Update w')
    %toc
    l = sum( max(0, 1 - Y.*(H(:,tridx)'*w)) ) + lambda1/2*w'*w
    %obj(Y, X, tridx, teidx, B, H, w, lambda1, lambda2, lambda3, lambda4, trXTX) 
    obj_value = obj(Y, X, tridx, teidx, B, H, w, lambda1, lambda2, lambda3, trXTX, loss, pre_obj)
    acc_tr = pred(Y, H(:,tridx), w)
    acc_tr_list(end+1) = acc_tr;
    acc_te = pred(Ytest, H(:,teidx), w)
    acc_te_list(end+1) = acc_te;
    if (pre_obj - obj_value)/pre_obj < tol,
	break;
    end
    pre_obj = obj_value;
    %acc_tr = pred(Y, H(:,tridx), w)
    %acc_tr_list(end+1) = acc_tr;
    %fprintf('training pred accuracy: %d\n', acc_tr)
    %acc_te = pred(Ytest, H(:,teidx), w)
    %acc_te_list(end+1) = acc_te;
    %fprintf('testing pred accuracy: %d\n', acc_te)
end
fprintf('final iteration: %d\n', iter);


end

