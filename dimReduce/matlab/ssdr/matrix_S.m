filepath = '../testdata/rcv1/';
split = '0.1/';
[label,feature]= libsvmread([filepath split 'trtr']);
[vl,vf]= libsvmread([filepath split 'vtr']);
[el,ef]= libsvmread([filepath split 'te']);
totall = [label;vl;el;];
maxf = max([size(feature,2),size(vf,2),size(ef,2)]);
feature(size(feature,1),maxf)=0;
vf(size(vf,1),maxf)=0;
ef(size(ef,1),maxf)=0;
totalf = [feature' vf' ef'];
alpha = 1;
beta = 20;
n = size(totall,1);
m = size(label,1);
S = ones(n)/(n*n);
nc = 0;
nm = 0;
for i=1:m
  for j=i+1:m
    if label(i)==label(j)
      nm = nm+1;
    else
      nc = nc+1;
    end 
  end
end


for i=1:m
  for j=i+1:m
    if label(i)==label(j)
      S(i,j) = S(i,j) + alpha/nc;
    else
      S(i,j) = S(i,j) - beta/nm;
    end 
  end
end

S = S + S';
D = diag(sum(S,2));
L = D - S;
L = sparse(L);
% rank k
k = 10; 
%W = randn(size(totalf,1),k);
%XLXT = totalf*L*totalf';
%[V,D] = eigs(XLXT);
