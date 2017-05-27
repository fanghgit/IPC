function y=AXZ(Z)
	global X D M1 M2 C1 C2 O1
	y1= X'*Z;
	y2 = D * y1;
	m = size(M1,2) + size(C1,2);
	n = m + size(O1,2);
	% get S*y1
        y3 = zeros(size(y1,1),1);
	y3(1:size(M1,1))= M1(1,1)*sum(y1(1:size(M1,2))) + C1(1,1)*sum(y1(size(M1,2)+1:m)) + O1(1,1)*sum(y1(m+1:end));
	y3(size(M1,1)+1:m)= C2(1,1)*sum(y1(1:size(C2,2))) + M2(1,1)*sum(y1(size(C2,2)+1:m)) + O1(1,1)*sum(y1(m+1:end));
	y3(n-m:end)=O1(1,1)*sum(y1); 
	y4 = y2 - y3; 
	%
	y = X * y4 ;
end
