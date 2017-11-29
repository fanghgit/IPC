using StatsBase

function main(input_path, output_path)
    X = readdlm(string(input_path, "/ratings.dat"), ',');
    X = ceil(Int64, X);
    n = size(X)[1];
    X = X[:,1:3];
    X[:,3] = ones(n);


    U = X[:,1];
    I = X[:,2];
    R = X[:,3];
    n = maximum(U); m = maximum(I);
    X = sparse(U, I, R, n, m);
    train = X';

    rows = rowvals(train);
    vals = nonzeros(train);
    start = 0;

    srand(1234)
    n_neg = 100;
    #user_test = [1:n];
    n_vt = Int(n_neg+1);
    item_test = zeros(n_vt*n);
    vals_test = -ones(n_vt*n);

    #user_validate = [1:n];
    item_validate = zeros(n_vt*n);
    vals_validate = -ones(n_vt*n);

    cols = zeros(nnz(train));
    cols_test = zeros(n_vt*n);
    cols_validate = zeros(n_vt*n);
    cc = 0;
    cc_test = 0;
    cc_validate = 0;
    for i = 1:n
        rg = nzrange(train, i);
        idx = rows[rg];
        len = length(idx);
        neg_idx = setdiff([1:m;], idx);

        if len <= 2
            println("len <= 2, error!");
        end

        (drop_idx_test, drop_idx_validate) = sample([1:len;], 2, replace = false);
        #construct test and validation sets
        item_test[Int(n_vt*(i-1)+1)] = idx[drop_idx_test];
        item_validate[Int(n_vt*(i-1)+1)] = idx[drop_idx_validate];
        vals_test[Int(n_vt*(i-1)+1)] = 1;
        vals_validate[Int(n_vt*(i-1)+1)] = 1;

        neg_sample_test = sample(neg_idx, n_neg, replace = false);
        neg_sample_validate = sample(neg_idx, n_neg, replace = false);
        item_test[ Int(n_vt*(i-1) + 2):Int(n_vt*i) ] = neg_sample_test;
        item_validate[ Int(n_vt*(i-1) + 2):Int(n_vt*i) ] = neg_sample_validate;
        vals_test[ Int(n_vt*(i-1) + 2):Int(n_vt*i) ] = -ones(n_neg);
        vals_validate[ Int(n_vt*(i-1) + 2):Int(n_vt*i) ] = -ones(n_neg);

        drop_idx_test += start;
        drop_idx_validate += start;
        start += len;
        vals[drop_idx_test] = 0;
        vals[drop_idx_validate] = 0;
        #deleteat!(rows, drop_idx_test);
        #deleteat!(rows, drop_idx_validate);
        for j = 1:(len)
            cc += 1;
            cols[cc] = i;
        end
        for j = 1:Int(n_neg+1)
            cc_test += 1;
            cc_validate += 1;
            cols_test[cc_test] = i;
            cols_validate[cc_validate] = i;
        end
    end

    cols = cols[vals .!= 0];
    rows = rows[vals .!= 0];
    vals = vals[vals .!= 0];

    writedlm(string( output_path, "/train_positive.csv"), [cols rows vals], ',');
    writedlm(string( output_path, "/test.csv"), [cols_test item_test vals_test], ',');
    writedlm(string( output_path, "/validate.csv"), [cols_validate item_validate vals_validate], ',');

    println("complete split train validation and test!")

    # #construct negative samples
    rows = round(Int64, rows);
    cols = round(Int64, cols);
    train = sparse(rows, cols, vals, m, n);
    rows = rowvals(train);

    #max positive-negative ratio
    ratio = 10;


    for r = 1:ratio
        total_neg = Int( r*nnz(train) );
        rows_neg = zeros(total_neg);
        cols_neg = zeros(total_neg);
        vals_neg = -ones(total_neg);
        cc = 0;
        start = 0;
        for i = 1:n
            rg = nzrange(train, i);
            idx = rows[rg];
            len = length(idx);
            neg_idx = setdiff([1:m;], idx);
            neg_sample = sample(neg_idx, Int(r*len), replace = false);
            rows_neg[ (start+1):Int(start+r*len) ] = neg_sample;
            start += r*len;

            for j = 1:Int(r*len)
                cc += 1;
                cols_neg[cc] = i;
            end
        end
        writedlm(string( output_path, "/negsamples", string(r), ".csv" ), [cols_neg rows_neg vals_neg], ',');
   	println("ratio: ", r, " complete!"); 
   end





end
