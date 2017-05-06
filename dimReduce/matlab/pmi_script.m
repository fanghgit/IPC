data = load('testdata/pmi.m');
PMI = data.M;
PPMI = max(0, log(PMI));

words_dict = containers.Map;
context_dict = containers.Map;

fid = fopen('testdata/pmi.words.vocab');
line = fgetl(fid);
i = 1;
while ischar(line),
    words_dict( line ) = i;
    line = fgetl(fid)
    i = i + 1;	
end

fid = fopen('testdata/pmi.contexts.vocab');
line = fgetl(fid);
i = 1;
while ischar(line),
    context_dict( line ) = i;
    line = fgetl(fid)
    i = i + 1;
end

pos_words = [];
fid = fopen('testdata/positive-words.txt');
line = fgetl(fid);
while ischar(line),
    if( line(1) == ';' ),
	continue;
    end
    pos_words(end+1) = line
end

neg_words = cell(0);
fid = fopen('testdata/negative-words.txt');
line = fgetl(fid);
while ischar(line),
    if( length(line) == 0 || line(1) == ';' ),
        line = fgetl(fid);
        continue;
    end
    neg_words{end+1} = line;
    line = fgetl(fid);
end

pos_idx = [];
for i = 1:size(pos_words, 2),
  if isKey(words_dict, pos_words{i}),
    pos_idx(end+1) = words_dict( pos_words{i} );
  end
end

neg_idx = [];
for i = 1:size(neg_words, 2),
  if isKey(words_dict, neg_words{i}),
    neg_idx(end+1) = words_dict( neg_words{i} );
  end
end

%delete common word
com = intersect(pos_idx, neg_idx);
com_idx = find(neg_idx == com);
neg_idx(com_idx) = [];

n = size(PPMI, 1);
dataidx = [pos_idx, neg_idx];
%teidx = [1:n];
%teidx[tridx] = [];

n_pos = length(pos_idx);
n_neg = length(neg_idx);
ndata = length(dataidx);
label = ones(ndata,1);
label(n_pos+1:ndata) = -1;

%split train test
perm =  randperm(ndata);
ratio = 0.8;
tridx = dataidx(perm(1:floor(ratio*ndata)));
teidx = dataidx(perm( floor(ratio*ndata)+1:ndata ));
train_label = label(perm(1:floor(ratio*ndata)));
test_label = label(perm( floor(ratio*ndata)+1:ndata ));

k = 10;
[U,S,V] = svds(PPMI, k);
train_inst = U(tridx,:)*sqrt(S);
test_inst = U(teidx,:)*sqrt(S);
model = train(train_label, train_inst, '-c 1');
[predict_label, accuracy, dec_values] = predict(test_label, test_inst, model); % test the training data


%k = 10;
%lambda1 = 1e-1;
%lambda2 = 1e-1;
%lambda3 = 1e-1;
%lambda4 = 1e-1;
%maxiter = 10;

%[ B, H, w, acc_tr, acc_te ] = drsvm( train_label, PPMI, tridx, teidx, k, lambda1, lambda2, lambda3, lambda4, maxiter, test_label );

expresult = [];

l_range = [1e-2,1e-1,1,1e1,1e2,1e3];
for l1 = l_range,
  for l2 = l_range,
    for l3 = l_range,
      [ B, H, w, acc_tr, acc_te ] = drsvm( train_label, data, k, l1, l2, l3, l3, maxiter, test_label );
      expresult(end+1,:) = [l1,l2,l3,l3,accuracy(1),acc_tr(end),acc_tr(end)];
    end
  end
end

cHeader = {lambda1' 'lambda2' 'lambda3' 'lambda4' 'svm_acc_te' 'acc_tr' 'acc_te'}; %dummy header
commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
commaHeader = commaHeader(:)';
textHeader = cell2mat(commaHeader); %cHeader in text with commas

fid = fopen('experimental_result/PPMIk=5.csv','w');
fprintf(fid,'%s\n',textHeader);
fclose(fid);
dlmwrite('experimental_result/PPMIk=5.csv',expresult,'-append');




