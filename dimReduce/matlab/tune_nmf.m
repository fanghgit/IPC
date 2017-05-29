ratio_list = [0.8, 0.4, 0.2, 0.1, 0.05, 0.01];
%p_list = [0, 0.25, 0.5, 0.75, 1];
l_list = [0.01, 0.1 , 1, 10, 1e2, 1e3, 1e4];
C_list = [0.01, 0.1, 1, 10 , 100];
file = {'rcv1', 'news20', 'real_sim', 'gis'};
expresult = []

for f = file,
        f = f{1};
	for ratio = ratio_list,
		path = ['testdata2/' f '/' num2str(ratio)];
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
		data = [trtr_inst; vtr_inst; te_inst];	

		k = 10;
		%[U,S,V] = svds(data, k);
		for l = l_list,
			tol = 1e-4;
			maxiter = 50;
			[H, B] = fasthals(data, k, l, l, tol, maxiter);
			drtr_inst = H(1:ntrtr,:);
    			drvtr_inst = H(ntrtr+1:ntr,:);
    			drte_inst = H(ntr+1:n,:);
			for C = C_list,
				par = ['-c ' num2str(C) ' -s 3'];
				model = train(trtr_label, sparse(drtr_inst), par, zeros(k, 1));
				[predict_label, drvtr_acc, dec_values] = predict(vtr_label, sparse(drvtr_inst), model);
    				[predict_label, drte_acc, dec_values] = predict(te_label, sparse(drte_inst), model);
				expresult(end+1,:) = [ratio, l, C, drvtr_acc(1), drte_acc(1)];
			end
		end
	end
	cHeader = {'ratio' 'l' 'C' 'acc_vtr' 'acc_tr'}; %dummy header
	commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
	commaHeader = commaHeader(:)';
	textHeader = cell2mat(commaHeader); %cHeader in text with commas
	
	f_path = ['4thexperimental_result/tune_nmf_svm/' f '.csv']
	fid = fopen(f_path,'w');
	fprintf(fid,'%s\n',textHeader);
	fclose(fid);
	dlmwrite(f_path,expresult,'-append');
	expresult = [];
end
