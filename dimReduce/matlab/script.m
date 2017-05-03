[label_vector, instance_matrix] = libsvmread('testdata/rcv1_train.binary'); 
n = size(label_vector, 1);
perm = randperm(n);
ratio = 0.01;
train_label = label_vector(perm(1:floor(n*ratio)));
train_inst = instance_matrix(perm(1:floor(n*ratio)), :);
test_label = label_vector(perm(floor(n*ratio+1):n));
test_inst = instance_matrix(perm(floor(n*ratio+1):n), :);

% model = train(train_label, train_inst, '-c 1');
% [predict_label, accuracy, dec_values] = predict(train_label, train_inst, model); % test the training data
% [predict_label, accuracy, dec_values] = predict(test_label, test_inst, model); % test the training data

% data = [train_inst; test_inst]';
% k = 10;
% scale = 1e1;
% lambda1 = scale*1e2;
% lambda2 = scale*1e3;
% lambda3 = scale*1;
% lambda4 = scale*1e1;
% maxiter = 50;
% [ B, H, w, acc_tr, acc_te ] = drsvm( train_label, data, k, lambda1, lambda2, lambda3, lambda4, maxiter, test_label );
% %pred(test_label, test_inst', )
% 
% plot([1:50], acc_tr, [1:50], acc_te)

ratio_list = [0.8, 0.1, 0.5, 0.01];
l_range = [1,1e1,1e2,1e3,1e4];
expresult = [];

for ratio = ratio_list,
    train_label = label_vector(perm(1:floor(n*ratio)));
    train_inst = instance_matrix(perm(1:floor(n*ratio)), :);
    test_label = label_vector(perm(floor(n*ratio+1):n));
    test_inst = instance_matrix(perm(floor(n*ratio+1):n), :);
    model = train(train_label, train_inst, '-c 1');
    [predict_label, accuracy, dec_values] = predict(test_label, test_inst, model); % test the training data
    data = [train_inst; test_inst]';
    k = 10;
    maxiter = 50;
    for l1 = l_range,
        for l2 = l_range,
            for l3 = l_range,
               for l4 = l_range,
                   [ B, H, w, acc_tr, acc_te ] = drsvm( train_label, data, k, l1, l2, l3, l4, maxiter, test_label );
                    expresult(end+1,:) = [ratio,l1,l2,l3,l4,accuracy(1),acc_tr(end),acc_tr(end)];
               end
            end
        end
    end
end


cHeader = {'ratio' 'lambda1' 'lambda2' 'lambda3' 'lambda4' 'svm_acc_te' 'acc_tr' 'acc_te'}; %dummy header
commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
commaHeader = commaHeader(:)';
textHeader = cell2mat(commaHeader); %cHeader in text with commas

fid = fopen('experimental_result/rcv1.csv','w');
fprintf(fid,'%s\n',textHeader);
fclose(fid);
dlmwrite('experimental_result/rcv1.csv',expresult,'-append');



