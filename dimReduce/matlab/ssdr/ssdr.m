global X m n none nnone m1 c1 o
addpath PROPACK
expresult=[];
filepath = '../testdata/webspam/';
ratio_list=[0.01,0.05,0.1,0.2,0.4,0.8];
%parpool(4)
for ind = 1:6
	ratio = ratio_list(ind)
[label,feature]= libsvmread([filepath num2str(ratio) '/' 'trtr']);
[vl,vf]= libsvmread([filepath num2str(ratio) '/' 'vtr']);
[el,ef]= libsvmread([filepath num2str(ratio) '/' 'te']);
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
% rank r
r = 10;
[V,D] = eigs('AXZ',size(X,1),r);
%W = randn(size(totalf,1),k);
%XLXT = totalf*L*totalf';
%[V,D] = eigs(XLXT);
%for index = 1:4
newf = sparse(feature*V);
nvf = sparse(vf*V);
nef = sparse(ef*V);
%newfp = py.numpy.reshape(newf(:)',int32(size(newf)),'F');
%nvfp = py.numpy.reshape(nvf(:)',int32(size(nvf)),'F');
%nep = py.numpy.reshape(nef(:)',int32(size(nef)),'F');
%labelp = py.list(label');
%elp = py.list(el');
%neigh = py.sklearn.neighbors.KNeighborsClassifier(int32(nn(1)));
%neigh.fit(newfp,labelp);
%keyboard;
libsvmwrite([filepath num2str(ratio) '/' 'newtf' '_' num2str(i) '_' num2str(j)],label,newf);
libsvmwrite([filepath num2str(ratio) '/' 'vtf' '_' num2str(i) '_' num2str(j)],vl,nvf);
libsvmwrite([filepath num2str(ratio) '/' 'tef' '_' num2str(i) '_' num2str(j)],el,nef);





%nn = [1,2,3,4];
%for index=1:4
%Mdl = fitcknn(newf,label,'NumNeighbors',nn(index));
%cor =0;
%ecor = 0;
%nvf = vf*V;
%nef = ef*V;
%for k=1:size(vf,1)
%	if (predict(Mdl,nvf(k,:))== vl(k))
%		cor=cor+1;
%	end
%end
%for k=1:size(ef,1)
%	if (predict(Mdl,nef(k,:))== el(k))
%		ecor = ecor +1;
%	end
%end
%acc = cor/size(vf,1);
%teacc = ecor / size(ef,1);
%model = train(full(label), sparse(newf));
%[~,acc,~] = predict(vl,sparse(vf*V),model);
%[~,teacc,~] = predict(el,sparse(ef*V),model);
%expresult(end+1,:) = [ratio,alpha,beta,acc(1),teacc(1)];
end
end
end
