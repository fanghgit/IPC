global X m n none nnone m1 c1 o
addpath PROPACK
filepath = '../testdata/rcv1/';
split = '0.8/';
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
n = size(totall,1);
m = size(label,1);
none = sum((label==1));
nnone = m - none;
nm = none*(none-1)/2 + nnone*(nnone-1)/2;
nc = none*nnone;
no = n - m;

X = totalf;

alphal = [0.1,1,10,100];
betal = [0.2,2,20,200];
accl = zeros(4,4);
teaccl = zeros(4,4);

for i=1:4
i 
for j=1:4
alpha = alphal(i);
beta = betal(j);
%M1 = ones(none,none)*(1/(n*n)-beta/nm);
%C1 = ones(none,nnone)*(1/(n*n)+alpha/nc);
%M2 = ones(nnone,nnone)*(1/(n*n)-beta/nm);
%C2 = ones(nnone,none)*(1/(n*n)+alpha/nc);
%O1 = ones(m,n-m)/(n*n);
%O2 = ones(n-m,m)/(n*n);
%O3 = ones(n-m,n-m)/(n*n);
m1 = 1/(n*n) - beta/nm;
c1 = 1/(n*n) + alpha/nc;
o = 1/(n*n);

%S1 = [M1 C1;M2 C2];
%S = [S1 O1; O2 O3];

%D = diag(sum(S,2));
%D = sparse(D);

%X = totalf;

%L = D - S;
%L = sparse(L);
% rank k
k = 10;
[V,D] = eigs('AXZ',size(X,1),k);
%W = randn(size(totalf,1),k);
%XLXT = totalf*L*totalf';
%[V,D] = eigs(XLXT);
newf = feature*V;
model = train(full(label), sparse(newf));
[~,acc,~] = predict(vl,sparse(vf*V),model);
[~,teacc,~] = predict(el,sparse(ef*V),model);
accl(i,j)=acc(1);
teaccl(i,j)=teacc(1);
end
end
save('r0.8','accl','teaccl');
