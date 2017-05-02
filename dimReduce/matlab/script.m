[label_vector, instance_matrix] = libsvmread('testdata/news20.binary'); 
n = size(label_vector, 1);
perm = randperm(n);
ratio = 0.05;
train_label = label_vector(perm(1:floor(n*ratio)));
train_inst = instance_matrix(perm(1:floor(n*ratio)), :);
test_label = label_vector(perm(floor(n*ratio+1):n));
test_inst = instance_matrix(perm(floor(n*ratio+1):n), :);

model = train(train_label, train_inst, '-c 1');
[predict_label, accuracy, dec_values] = predict(train_label, train_inst, model); % test the training data
[predict_label, accuracy, dec_values] = predict(test_label, test_inst, model); % test the training data

%SVMModel = fitcsvm(H',Y,'KernelFunction','linear');


data = [train_inst; test_inst]';
k = 10;
lambda1 = 1;
lambda2 = 1e-3;
maxiter = 50;
[ B, H, w ] = drsvm( train_label, data, k, lambda1, lambda2, maxiter, test_label );
%pred(test_label, test_inst', )




