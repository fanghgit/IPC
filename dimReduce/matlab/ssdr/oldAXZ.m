function y=AXZ(Z)
	global X m n none nnone m1 c1 o 
	y1= X'*Z;
	D = zeros(n,1);
	D(1:none)= m1*none + c1 *nnone + o*(n-m);
	D(none+1:m) = c1 * none + m1 * nnone + o*(n-m);
	D(m+1:end) = o*n;
	y2 = D .* y1;
	%m = size(M1,2) + size(C1,2);
	%n = m + size(O1,2);
	% get S*y1
        y3 = zeros(size(y1,1),1);
	y3(1:none)= m1*sum(y1(1:none)) + c1*sum(y1(none+1:m)) + o*sum(y1(m+1:end));
	y3(none+1:m)= c1*sum(y1(1:none)) + m1*sum(y1(none+1:m)) + o*sum(y1(m+1:end));
	y3(m+1:end)=o*sum(y1); 
	y4 = y2 - y3; 
	%
	y = X * y4 ;
end
