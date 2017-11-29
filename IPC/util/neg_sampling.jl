using StatsBase

function main(input_path, output_path, ratio)
    #input_path = ARGS[1];
    #output_path = ARGS[1];
    #ratio = ARGS[2];
    X = readdlm(string(input_path, "/train_positive.csv"), ',');
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
            n_neg_sample = min(Int(r*len), length(neg_idx));
            if r*len > length(neg_idx)
                println("a lot of ratings? user: ", i, ", ", len, " ratings")
            end
            neg_sample = sample(neg_idx, n_neg_sample, replace = false);
            rows_neg[ (start+1):Int(start+n_neg_sample) ] = neg_sample;
            start += n_neg_sample;

            for j = 1:n_neg_sample
                cc += 1;
                cols_neg[cc] = i;
            end
        end
        rows_neg = rows_neg[1:start];
        cols_neg = cols_neg[1:start];
        vals_neg = vals_neg[1:start];
        writedlm(string( output_path, "/negsamples", string(r), ".csv" ), [cols_neg rows_neg vals_neg], ',');
        println("ratio: ", r, " complete!");
    end


end
