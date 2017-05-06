[label_vector, instance_matrix] = libsvmread('testdata/rcv1_train.binary');
n = size(label_vector, 1);
perm = randperm(n);
ratio = 0.8;
train_label = label_vector(perm(1:floor(n*ratio)));
train_inst = instance_matrix(perm(1:floor(n*ratio)), :);
test_label = label_vector(perm(floor(n*ratio+1):n));
test_inst = instance_matrix(perm(floor(n*ratio+1):n), :);

data = [train_inst; test_inst]';
k = 10;
scale = 1e-2;
lambda1 = scale*1;
lambda2 = scale*1;
lambda3 = scale*1;
lambda4 = scale*1;
maxiter = 50;
ntr = size(train_label, 1);
tridx = [1:ntr];
teidx = [ntr+1:n];

[ B, H, w, acc_tr, acc_te ] = drsvm( train_label, data, tridx, teidx, k, lambda1, lambda2, lambda3, lambda4, maxiter, test_label );


