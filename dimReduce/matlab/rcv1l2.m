%[label_vector, instance_matrix] = libsvmread('testdata/rcv1_train.binary');
%[label_vector2, instance_matrix2] = libsvmread('testdata/rcv1_test.binary');
%label_vector = [label_vector; label_vector2];
%size(label_vector)
%instance_matrix = [instance_matrix; instance_matrix2];
%size(instance_matrix)
%n = size(label_vector, 1);
%[n, m] = size(instance_matrix);
%perm = randperm(n);
%ratio = 0.01;
%train_label = label_vector(perm(1:floor(n*ratio)));
%train_inst = instance_matrix(perm(1:floor(n*ratio)), :);
%test_label = label_vector(perm(floor(n*ratio+1):n));
%test_inst = instance_matrix(perm(floor(n*ratio+1):n), :);

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

%ratio_list = [0.8, 0.4, 0.2, 0.1];
ratio_list = [0.05, 0.01];
l_range = [1,1e1,1e2,1e3];
expresult = [];

for ratio = ratio_list,
    %perm = randperm(n);
    %ntr = floor(n*ratio);
    %tr = perm(1:ntr);
    %te = perm(ntr+1:n);
    %perm2 = randperm(ntr);
    %ntrtr = floor(ntr*ratio);
    %trtr = tr( perm2(1:ntrtr) );
    %vtr = tr( perm2(ntrtr+1:ntr) );
	
    %trtr_label = label_vector(trtr);
    %trtr_inst = instance_matrix(trtr, :);
    %vtr_label = label_vector(vtr);
    %vtr_inst = instance_matrix(vtr, :);
    %tr_label = label_vector(tr);
    %tr_inst = instance_matrix(tr, :);
    %te_label = label_vector(te);
    %te_inst = instance_matrix(te, :);
    path = ['testdata2/rcv1/' num2str(ratio)];
    [trtr_label, trtr_inst] = libsvmread([path '/trtr']);
    [vtr_label, vtr_inst] = libsvmread([path '/vtr']);
    %[tr_label, tr_inst] = libsvmread([path '/tr']);
    [te_label, te_inst] = libsvmread([path '/te']); 
    [ntrtr, mtrtr] = size(trtr_inst);
    [nvtr, mvtr] = size(vtr_inst);
    [nte, mte] = size(te_inst);
    maxdim = max([mtrtr, mvtr, mte]);
    trtr_inst = [trtr_inst, zeros(ntrtr, max(0, maxdim - mtrtr))];
    vtr_inst = [vtr_inst, zeros(nvtr, max(0, maxdim - mvtr))];
    te_inst = [te_inst, zeros(nte, max(0, maxdim - mte))];
    tr_label = [trtr_label; vtr_label];
    tr_inst = [trtr_inst; vtr_inst];
    [ntr, mtr] = size(tr_inst);
    vtrte_label = [vtr_label; te_label];
    n = ntr + nte;
    %model = train(trtr_label, trtr_inst, '-c 1 -s 3', zeros(size(trtr_inst, 2),1) );
    %[predict_label, trtraccuracy, dec_values] = predict(vtr_label, vtr_inst, model); % test the training data
    model = train(tr_label, tr_inst, '-c 1 -s 1', zeros(size(tr_inst, 2), 1));
    [predict_label, teaccuracy, dec_values] = predict(te_label, te_inst, model); 
    data = [trtr_inst; vtr_inst; te_inst];
    %data = [tr_inst; te_inst];
    k = 10;
    %[U,S,V] = svds(trtrdata, k);
    %drtrtr_inst = U(1:ntrtr,:) * sqrt(S);
    %drvtr_inst = U(ntrtr+1:ntr,:) * sqrt(S);
    %model = train(trtr_label, sparse( drtrtr_inst ), '-c 1 -s 3', zeros(k, 1));
    %[predict_label, drvtraccuracy, dec_values] = predict(vtr_label, sparse( drvtr_inst ), model);     

    [U,S,V] = svds(data, k);
    drtr_inst = U(1:ntrtr,:) * sqrt(S);
    drvtr_inst = U(ntrtr+1:ntr,:) * sqrt(S);
    drte_inst = U(ntr+1:n,:) * sqrt(S);
    model = train(trtr_label, sparse(drtr_inst), '-c 1 -s 1', zeros(k, 1));
    [predict_label, drvtr_acc, dec_values] = predict(vtr_label, sparse(drvtr_inst), model);
    [predict_label, drte_acc, dec_values] = predict(te_label, sparse(drte_inst), model);

    k = 10;
    maxiter = 30;
    tol = 1e-3;
    loss = 'l2';
    nonneg = 1;
    vtrte_label = [vtr_label; te_label];
    for l1 = l_range,
        for l2 = l_range,
            for l3 = l_range,
               %for l4 = l_range,
               [ B, H, w, acc_trtr, acc_vtr ] = drsvm( trtr_label, data', [1:ntrtr], [ntrtr+1:n], k, l1, l2, l3, maxiter, tol, loss, nonneg, vtrte_label);
               vtr_acc = pred(vtr_label, H(:,ntrtr+1:ntr), w);
	       te_acc = pred(te_label, H(:,ntr+1:n), w);
		%[ B, H, w, acc_tr, acc_te ] = drsvm( tr_label, data', [1:ntr], [ntr+1:n], k, l1, l2, l3, maxiter, tol, loss, nonneg, te_label );
	       %expresult(end+1,:) = [ratio,l1,l2,l3, trtraccuracy(1), teaccuracy(1), drvtraccuracy(1), drteaccuracy(1), acc_trtr(end), acc_vtr(end), acc_tr(end), acc_te(end) ];
               expresult(end+1,:) = [ratio,l1,l2,l3, teaccuracy(1), drvtr_acc(1), drte_acc(1), acc_trtr(end), vtr_acc, te_acc ];
               %end
            end
        end
    end
end


cHeader = {'ratio' 'lambda1' 'lambda2' 'lambda3' 'svm_acc_te' 'dr_acc_vtr' 'dr_acc_te' 'acc_trtr' 'acc_vtr' 'acc_te'}; %dummy header
commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
commaHeader = commaHeader(:)';
textHeader = cell2mat(commaHeader); %cHeader in text with commas

%fid = fopen('4thexperimental_result/rcv1l2nn.csv','w');
%fprintf(fid,'%s\n',textHeader);
%fclose(fid);
dlmwrite('4thexperimental_result/rcv1l2nn.csv',expresult,'-append');
disp('complete!')


