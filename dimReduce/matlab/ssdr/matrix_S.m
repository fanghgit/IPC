global X D M1 M2 C1 C2 O1
addpath PROPACK
filepath = '../testdata/news20/';
split = '0.1/';
[label,feature]= libsvmread([filepath split 'trtr']);
[vl,vf]= libsvmread([filepath split 'vtr']);
[el,ef]= libsvmread([filepath split 'te']);
all=[label feature];
all = sortrows(all);
label = all(:,1);
feature = all(:,2:end);
totall = [label;vl;el];
maxf = max([size(feature,2),size(vf,2),size(ef,2)]);
feature(size(feature,1),maxf)=0;
vf(size(vf,1),maxf)=0;
ef(size(ef,1),maxf)=0;
totalf = [feature' vf' ef'];
alpha = 1;
beta = 20;
n = size(totall,1);
m = size(label,1);
none = sum((label==1));
nnone = m - none;
nm = none*(none-1)/2 + nnone*(nnone-1)/2;
nc = none*nnone;
no = n - m;

M1 = ones(none,none)*(1/(n*n)-beta/nm);
C1 = ones(none,nnone)*(1/(n*n)+alpha/nc);
M2 = ones(nnone,nnone)*(1/(n*n)-beta/nm);
C2 = ones(nnone,none)*(1/(n*n)+alpha/nc);
O1 = ones(m,n-m)/(n*n);
O2 = ones(n-m,m)/(n*n);
O3 = ones(n-m,n-m)/(n*n);

S1 = [M1 C1;M2 C2];
S = [S1 O1; O2 O3];

D = diag(sum(S,2));
D = sparse(D);
X = totalf;

%L = D - S;
%L = sparse(L);
% rank k
k = 10;
[V,D] = eigs('AXZ',size(X,1),k);
%W = randn(size(totalf,1),k);
%XLXT = totalf*L*totalf';
%[V,D] = eigs(XLXT);
