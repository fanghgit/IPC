[tl,tf] = libsvmread('testdata/news20/0.1/trtr');
[vl,vf] = libsvmread('testdata/news20/0.1/vtr');
tl( ~any(tf,2),:) = [];
vl( ~any(vf,2),:) = [];
tf( ~any(tf,2), : ) = [];  %rows
vf( ~any(vf,2), : ) = [];  %rows

maxd = max(size(tf,2),size(vf,2));
vf(size(vf,1),maxd)=0;
X = [tf' vf'];
X( ~any(X,2), : ) = [];

Y = zeros(2,size(tl,1));
for i=1:size(tl)
	if tl(i)==-1
		Y(1,i)=1;
	else
		Y(2,i)=1;
	end
end


tridx = [1:size(tl,1)];
teidx = [size(tl,1)+1:size(tl,1)+size(vl,1)];
[B,S] = SSNMF(X,Y,10,1, tridx , teidx, 1e-5, 50);
