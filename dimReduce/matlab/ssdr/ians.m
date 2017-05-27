filepath = '../testdata/news20/';
split = '0.1/';
[label,feature]= libsvmread([filepath split 'trtr']);
[vl,vf]= libsvmread([filepath split 'vtr']);
%[el,ef]= libsvmread([filepath split 'te']);
totall = [label;vl];
maxf = max([size(feature,2),size(vf,2)]);
feature(size(feature,1),maxf)=0;
vf(size(vf,1),maxf)=0;
%ef(size(ef,1),maxf)=0;
totalf = [feature' vf'];
n = size(totall,1);
m = size(label,1);
S = sparse(n,n);

for i=1:m
  for j=i+1:m
    if label(i)==label(j)
      S(i,j) = 1;
    else
      S(i,j) = -1;
    end 
  end
end

S = S + S'+speye(n);
D = diag(sum(S,2));
L = D - S;
L = sparse(L);
% rank k
k = 10; 
%W = randn(size(totalf,1),k);
%XLXT = totalf*L*totalf';
%[V,D] = eigs(XLXT);
