#implementation of BPR using full and subsample approach
using StatsBase
#

function comp_m(U, V, X, d1, d2, rows, vals, cols)

	mvals = zeros(nnz(X))
	cc=0
	for i=1:d1
		tmp = nzrange(X,i)
		d2_bar = rows[tmp];
		ui = U[:,i]
		for j in d2_bar
			cc+=1
			mvals[cc] = dot(ui, V[:,j])
		end
	end
	return sparse(rows, cols, mvals, d2, d1);

#	m = spzeros(d2,d1);
#	for i = 1:d1
#		tmp = nzrange(X, i)
#		d2_bar = rows[tmp];
#		ui = U[:, i]
#		for j in d2_bar
#			m[j,i] = dot(ui,V[:,j])
#		end
#	end
#	return m
end


function objective(m, U, V, X, d1, lambda, rows, vals, n_pairs_user, n_pairs_item)
	res1 = 0.0
	#res1 = lambda / 2.0 * (vecnorm(U) ^ 2 +vecnorm(V) ^ 2)
	#change regularization
	tmpU = sqrt(n_pairs_user') .* U;
	tmpV = sqrt(n_pairs_item') .* V;
	res1 = lambda / 2.0 * (vecnorm(tmpU) ^ 2 +vecnorm(tmpV) ^ 2)
	for i in 1:d1
		tmp = nzrange(X, i)
		d2_bar = rows[tmp];
		vals_d2_bar = vals[tmp];
		len = size(d2_bar)[1];
		mm = nonzeros(m[:,i])

		levels = sort(unique(vals_d2_bar))
		nlevels = size(levels)[1]
		perm_ind = sortperm(mm)
		mm_sorted = mm[perm_ind]
#		b_sorted = b[perm_ind]
		vals_sorted = vals_d2_bar[perm_ind]
		for j=1:len
			for k=1:nlevels
				if vals_sorted[j] == levels[k]
					vals_sorted[j] = k
					break
				end
			end
		end
#		d2bar_sorted = d2_bar[perm_ind]
#		allsum = zeros(nlevels)
#		for j=1:len
#			allsum[vals_sorted[j]] += b_sorted[j]
#		end


		nowleft = 1
#		nowright = 1
		countleft = zeros(Int, nlevels)
#		countright = zeros(Int, nlevels)
#		for j=1:len
#			countright[vals_sorted[j]] +=1
#		end

		nowleftsum = zeros(nlevels)
		nowleftsqsum = zeros(nlevels)
#		nowrightsum = allsum
#		cplist = zeros(len)
		for j = 1:len
			nowcut = mm_sorted[j]
			nowval = vals_sorted[j]
			while (nowleft <= len) && (mm_sorted[nowleft] < nowcut+1.0)
				nowleftsum[vals_sorted[nowleft]] += (mm_sorted[nowleft]-1)
				nowleftsqsum[vals_sorted[nowleft]] += ((mm_sorted[nowleft]-1)^2)
				countleft[vals_sorted[nowleft]] +=1
				nowleft+=1
			end
#			while (nowright <= len) && (mm_sorted[nowright] <= nowcut-1.0)
#				nowrightsum[vals_sorted[nowright]] -= b_sorted[nowright]
#				countright[vals_sorted[nowright]] -=1
#				nowright+=1
#			end

#			c_p = 0.0
#			for k=1:nowval-1
#				c_p += (countright[k]*b_sorted[j] - nowrightsum[k])
#			end
			for k=nowval+1:nlevels
				res1 += (countleft[k]*(nowcut^2)-2.0*nowcut*nowleftsum[k] + nowleftsqsum[k])
			end
		end
	end

#	res = 0.0
#	res = lambda / 2.0 * (vecnorm(U) ^ 2 +vecnorm(V) ^ 2)
#	for i in 1:d1
#		tmp = nzrange(X, i)
#		d2_bar = rows[tmp];
#		vals_d2_bar = vals[tmp];
#		len = size(d2_bar)[1];
#		mm = nonzeros(m[:,i])
#
#		for j in 1:(len - 1)
#			for k in (j + 1):len
#				if vals_d2_bar[j] == vals_d2_bar[k]
#					continue
#				elseif vals_d2_bar[j] > vals_d2_bar[k]
#					y_ipq = 1.0
#				elseif vals_d2_bar[k] > vals_d2_bar[j]
#					y_ipq = -1.0
#				end
#				mask = y_ipq * (mm[j]-mm[k])
#				if mask < 1.0
#					res += (1.0 - mask) ^ 2
#				end
#			end
#		end
#	end
	return res1
end


function compute_pairwise_error(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, hr_k, ndcg_k)

    npairs = 0;
    correct = 0;
    hit = 0;
    ndcg_sum = 0.;
    for i = 1:d1
        rg = nzrange(Y, i);
        idx = rows_t[rg];
        len = length(idx);
        vals_tmp = vals_t[rg];
        npairs += len;
        pos_idx = 0;
        for j = 1:len
            if vals_tmp[j] == 1
                pos_idx = idx[j];
                break
            end
        end
        if pos_idx == 0
            error("can not find positive example")
        end
        neg_idx = setdiff(idx, pos_idx);
        for j in neg_idx
            if dot( U[:,i], V[:,pos_idx] - V[:,j]) > 0
                correct += 1
            end
        end
        #calculate hit ratio
        m = vec( (U[:,i])' * V[:,idx] );
        score = dot(U[:,i], V[:,pos_idx]);
        sorted_m = sort(m, rev = true);
        if score > sorted_m[hr_k]
            hit += 1;
        end
        #caculate nDCG
        vals_tmp[vals_tmp .== -1] = 0;
        p1 = sortperm(m, rev = true)
		p1 = p1[1:ndcg_k]
		M1 = vals_tmp[p1]
		p2 = sortperm(vals_tmp, rev = true)
		p2 = p2[1:ndcg_k]
		M2 = vals_tmp[p2]
		dcg = 0.; dcg_max = 0.
		for k = 1:ndcg_k
			dcg += (2 ^ M1[k] - 1) / log2(k + 1)
			dcg_max += (2 ^ M2[k] - 1) / log2(k + 1)
		end
		ndcg_sum += dcg / dcg_max
    end

    pairwise_accuracy = correct/npairs;
    hr = hit/d1;
    ndcg = ndcg_sum/d1;

    return pairwise_accuracy, hr, ndcg

end

function precompute_ss(X, d1, d2, prob, rows, cols, vals, epochSize)
    #s_cols = sample([1:d1;], WeightVec(prob), Int64(epochSize));
    s_cols = sample([1:d1;], Int64(epochSize));
    ss = []
    for i in s_cols
        rg = nzrange(X, i);
        idx = rowvals(X)[rg];
        len = length(idx);
        #total_neg_idx = setdiff([1:d2;], idx);
        j0 = rand(1:len);
        neg_idx = rand(1:d2);
        while neg_idx in idx
            neg_idx = rand(1:d2);
        end
        pos_idx = idx[j0];
        #neg_idx = total_neg_idx[k0];
        push!(ss, (i, pos_idx, neg_idx));
    end
    return ss
end

#BPR_full("data/ml-1m/train_positive.csv", "data/ml-1m/test.csv", 100, 0.01, 0, 0.01)
#BPR_full("data/sub_ml-1m/train_positive.csv", "data/sub_ml-1m/test.csv", 100, 0.1, 0, 0.01)
#BPR_full("data/ml-10M100K/train_positive.csv", "data/ml-10M100K/test.csv", 100, 7000, 0, 0.01)
function BPR_full(train, test, r, lambda, epochSize, stepsize)
    X = readdlm(train, ',' );
	x = vec(round( Int64, X[:,1]) );
	y = vec(round( Int64, X[:,2]) );
	v = vec(round( Int64, X[:,3]) );
	Y = readdlm(test, ',' );
	xx = vec(round( Int64, Y[:,1]) );
	yy = vec(round( Int64, Y[:,2]) );
	vv = vec(round( Int64, Y[:,3]) );

    n = max(maximum(x), maximum(xx)); msize = max(maximum(y), maximum(yy));
    X = sparse(x, y, v, n, msize); # userid by movieid
    Y = sparse(xx, yy, vv, n, msize);
    # julia column major
    # now moveid by userid
    X = X';
    Y = Y';

    rows = rowvals(X);
    vals = nonzeros(X);
    cols = zeros(Int, size(vals)[1]);

    d2, d1 = size(X);
    cc = 0;
    for i = 1:d1
        tmp = nzrange(X, i);
        nowlen = size(tmp)[1];
        for j = 1:nowlen
            cc += 1
            cols[cc] = i
        end
    end

    rows_t = rowvals(Y);
    vals_t = nonzeros(Y);
    cols_t = zeros(Int, size(vals_t)[1]);
    cc = 0;
    for i = 1:d1
        tmp = nzrange(Y, i);
        nowlen = size(tmp)[1];
        for j = 1:nowlen
            cc += 1
            cols_t[cc] = i
        end
    end

    npairs = 0;
    n_pairs_user = zeros(d1);
    n_pairs_item = zeros(d2);
    for i = 1:d1
        tmp = nzrange(X, i);
        nowlen = size(tmp)[1];
        idx = rows[tmp];
        npairs += nowlen*(d2-nowlen);
        n_pairs_user[i] = nowlen*(d2 - nowlen);
        n_pairs_item += nowlen
        n_pairs_item[idx] += (d2 - nowlen - nowlen)
    end
    prob = n_pairs_user / sum(n_pairs_user);
    @printf("number of training pairs: %i \n", npairs)
    println(var(prob))

    hr_k = 10;
    ndcg_k = 10;
	# initialize U, V
	srand(1234)
	U = 0.1*randn(r, d1); V = 0.1*randn(r, d2);

    totaltime = 0.00000;
    println("iter time pairwise_error")

    pairwise_error, hr, ndcg = compute_pairwise_error(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, hr_k, ndcg_k)
    println("[", 0, ", ", totaltime, ", ", pairwise_error, ", ", hr, ", ", ndcg, "]")

    maxEpoch = 50;
    for iter in 1:maxEpoch
        epochSize = nnz(X);
        tic();
        ss = precompute_ss(X, d1, d2, prob, rows, cols, vals, epochSize);
        tt = toq();
        totaltime += tt;
        @printf("time spend on sampling: %f \n", tt)
        tic();

        for (i, j, k) in ss
            #rg = nzrange(X, i);
			#idx = rowvals(X)[rg];
			#len = length(idx);
            ui = U[:,i];
			#tmp_val = vals[rg];
            vj = V[:,j];
            vk = V[:,k];

            a_ijk = 1.
            mask = a_ijk * ( ui'*(vj - vk) );
            aaa = 2*min(mask-1, 0)*a_ijk;
            # grad_vj = (aaa .* ui + lambda/n_pairs_item[j] .* vj);
			# grad_vk = (-aaa .* ui + lambda/n_pairs_item[k] .* vk);
			# grad_ui = (aaa .* (vj - vk) + lambda/n_pairs_user[i] .* ui);

            grad_vj = (aaa .* ui + lambda .* vj);
            grad_vk = (-aaa .* ui + lambda .* vk);
            grad_ui = (aaa .* (vj - vk) + lambda .* ui);

            V[:,j] -= stepsize * grad_vj;
            V[:,k] -= stepsize * grad_vk;
            U[:,i] -= stepsize * grad_ui;


        end

        totaltime += toq();
        pairwise_error, hr, ndcg = compute_pairwise_error(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, hr_k, ndcg_k)
        println("[", 0, ", ", totaltime, ", ", pairwise_error, ", ", hr, ", ", ndcg, "]")

    end


end


function precompute_subsample_ss(X_pos, X_neg, d1, d2, prob, rows_pos, rows_neg, epochSize)
    #s_cols = sample([1:d1;], WeightVec(prob), Int64(epochSize));
    s_cols = sample([1:d1;], Int64(epochSize));
    ss = []
    for i in s_cols
        rg_pos = nzrange(X_pos, i);
        rg_neg = nzrange(X_neg, i);
        idx_pos = rows_pos[rg_pos];
        idx_neg = rows_neg[rg_neg];
        j = rand(idx_pos);
        k = rand(idx_neg);

        push!(ss, (i, j, k));
    end
    return ss
end

#ml-1m: lambda = 0.01
#BPR_subsample("data/ml-1m/train_positive.csv", "data/ml-1m/negsamples1.csv", "data/ml-1m/test.csv", 100, 0.01, 0, 0.01)
#BPR_subsample("data/sub_ml-1m/train_positive.csv", "data/sub_ml-1m/negsamples1.csv", "data/sub_ml-1m/test.csv", 100, 0.1, 0, 0.01)

function BPR_subsample(train_positive, train_negative, test, r, lambda, epochSize, stepsize)
    X_pos = readdlm(train_positive, ',' );
    xpos = vec( round(Int64, X_pos[:,1]) );
	ypos = vec( round(Int64, X_pos[:,2]) );
	vpos = vec( round(Int64, X_pos[:,3]) );
    X_neg = readdlm(train_negative, ',' );
    xneg = vec( round(Int64, X_neg[:,1]) );
	yneg = vec( round(Int64, X_neg[:,2]) );
	vneg = vec( round(Int64, X_neg[:,3]) );
    Y = readdlm(test, ',');
    xx = vec( round(Int64, Y[:,1]) );
	yy = vec( round(Int64, Y[:,2]) );
	vv = vec( round(Int64, Y[:,3]) );

    #construct X
    x = vcat(xpos, xneg);
	y = vcat(ypos, yneg);
	v = vcat(vpos, vneg);


    n = max(maximum(xpos), maximum(xneg), maximum(xx));
    msize = max(maximum(ypos), maximum(yneg), maximum(yy));

    X_pos = sparse(xpos, ypos, vpos, n, msize); # userid by movieid
    X_neg = sparse(xneg, yneg, vneg, n, msize);
    X = sparse(x, y, v, n, msize);
    Y = sparse(xx, yy, vv, n, msize);
    # julia column major
    # now moveid by userid
    X_pos = X_pos';
    X_neg = X_neg';
    X = X';
    Y = Y';

    rows = rowvals(X);
	vals = nonzeros(X);
	cols = zeros(Int, size(vals)[1]);

	d2, d1 = size(X);
	cc = 0;
	for i = 1:d1
		tmp = nzrange(X, i);
		nowlen = size(tmp)[1];
		for j = 1:nowlen
			cc += 1
			cols[cc] = i
		end
	end

    #Get rows, vals and cols
    rows_pos = rowvals(X_pos);
    vals_pos = nonzeros(X_pos);
    cols_pos = zeros(Int, length(vals_pos));

    d2, d1 = size(X_pos);
    cc = 0;
    for i = 1:d1
        tmp = nzrange(X_pos, i);
        nowlen = size(tmp)[1];
        for j = 1:nowlen
            cc += 1
            cols_pos[cc] = i
        end
    end

    rows_neg = rowvals(X_neg);
    vals_neg = nonzeros(X_neg);
    cols_neg = zeros(Int, length(vals_neg));
    cc = 0;
    for i = 1:d1
        tmp = nzrange(X_neg, i);
        nowlen = size(tmp)[1];
        for j = 1:nowlen
            cc += 1
            cols_neg[cc] = i
        end
    end


    rows_t = rowvals(Y);
    vals_t = nonzeros(Y);
    cols_t = zeros(Int, size(vals_t)[1]);
    cc = 0;
    for i = 1:d1
        tmp = nzrange(Y, i);
        nowlen = size(tmp)[1];
        for j = 1:nowlen
            cc += 1
            cols_t[cc] = i
        end
    end

    # npairs = 0;
    # n_pairs_user = zeros(d1);
    # n_pairs_item = zeros(d2);
    # for i = 1:d1
    #     tmp = nzrange(X_pos, i);
    #     nowlen = size(tmp)[1];
    #     idx = rows_pos[tmp];
    #     npairs += nowlen*(d2-nowlen);
    #     n_pairs_user[i] = nowlen*(d2 - nowlen);
    #     n_pairs_item += nowlen
    #     n_pairs_item[idx] += (d2 - nowlen - nowlen)
    # end
    # prob = n_pairs_user / sum(n_pairs_user);
    # @printf("number of training pairs: %i\n", npairs)

    # avg lambda user: 0.020724
    # avg lambda item: 0.003563


    npairs = 0;
    n_pairs_user = zeros(Int, d1);
    n_pairs_item = zeros(Int, d2);
    for i = 1:d1
        tmp_pos = nzrange(X_pos, i);
        tmp_neg = nzrange(X_neg, i);
        len_pos = size(tmp_pos)[1];
        len_neg = size(tmp_neg)[1];

        idx_pos = rows_pos[tmp_pos];
        idx_neg = rows_neg[tmp_neg];

        npairs += len_pos*len_neg;
        n_pairs_user[i] = len_pos*len_neg;
        n_pairs_item[idx_pos] += len_neg;
        n_pairs_item[idx_neg] += len_pos;
    end
    prob = n_pairs_user / sum(n_pairs_user);
    @printf("number of training pairs: %i\n", npairs)
    println(var(prob))

    hr_k = 10;
    ndcg_k = 10;
	# initialize U, V
	srand(1234)
	U = 0.1*randn(r, d1); V = 0.1*randn(r, d2);

    totaltime = 0.00000;
    println("iter time pairwise_error")
    #Compute initial pairwise loss
    pairwise_error, hr, ndcg = compute_pairwise_error(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, hr_k, ndcg_k)
    m = comp_m(U, V, X, d1, d2, rows, vals, cols)
	nowobj = objective(m, U, V, X, d1, lambda, rows, vals, n_pairs_user, n_pairs_item)
    println("[", 0, ", ", totaltime, ", ", nowobj, ", ", pairwise_error, ", ", hr, ", ", ndcg, "]")

    maxEpoch = 50;
    for iter in 1:maxEpoch
        tic();
        #Set epochSize
        epochSize = nnz(X_pos);
        ss = precompute_subsample_ss(X_pos, X_neg, d1, d2, prob, rows_pos, rows_neg, epochSize)
        tt = toq();
        totaltime += tt;
        @printf("time spend on sampling: %f\n", tt)
        tic();
        for (i, j, k) in ss
            #rg_pos = nzrange(X_pos, i);
            #rg_neg = nzrange(X_neg, i);
            #len_pos = size(rg_pos)[1];
            #len_neg = size(rg_neg)[1];
            #idx_pos = rows_pos[rg_pos];
            #idx_neg = rows_neg[rg_neg];

            #j = rand(idx_pos);
            #k = rand(idx_neg);

            ui = U[:,i];
            #tmp_val = vals[rg];
            vj = V[:,j];
            vk = V[:,k];

            a_ijk = 1.
            mask = a_ijk * ( ui'*(vj - vk) );
            aaa = 2*min(mask-1, 0)*a_ijk;
            # grad_vj = (aaa .* ui + lambda/n_pairs_item[j] .* vj);
			# grad_vk = (-aaa .* ui + lambda/n_pairs_item[k] .* vk);
			# grad_ui = (aaa .* (vj - vk) + lambda/n_pairs_user[i] .* ui);

            grad_vj = (aaa .* ui + lambda .* vj);
            grad_vk = (-aaa .* ui + lambda .* vk);
            grad_ui = (aaa .* (vj - vk) + lambda .* ui);

            V[:,j] -= stepsize * grad_vj;
            V[:,k] -= stepsize * grad_vk;
            U[:,i] -= stepsize * grad_ui;

        end

        totaltime += toq();
        pairwise_error, hr, ndcg = compute_pairwise_error(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, hr_k, ndcg_k)
        m = comp_m(U, V, X, d1, d2, rows, vals, cols)
    	nowobj = objective(m, U, V, X, d1, lambda, rows, vals, n_pairs_user, n_pairs_item)
        println("[", 0, ", ", totaltime, ", ", nowobj, ", ",  pairwise_error, ", ", hr, ", ", ndcg, "]")

    end

end
