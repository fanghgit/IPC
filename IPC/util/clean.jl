function main(path)
    X = readdlm(string(path, "/ratings.dat"), ',')
    #X = ceil(Int64, X);
    n = size(X)[1];
    X = X[:,1:3];
    #X[:,3] = ones(n);

    U = round(Int64, X[:,1]);
    I = round(Int64, X[:,2]);
    R = X[:,3];
    n = maximum(U); m = maximum(I);
    X = sparse(U, I, R, n, m);
    println("number of users: ", n, ", number of items: ", m);
    train = X';
    rows = rowvals(train);
    vals = R;
    cols = zeros(Int, nnz(train));

    bad_users = [];

    cc = 0;
    for i = 1:n
        rg = nzrange(train, i);
        idx = rows[rg];
        len = length(idx);
        if(len < 20)
            vals[rg] = 0;
            push!(bad_users, i);
        end
        for j = 1:len
            cc += 1;
            cols[cc] = i;
        end
    end
    train_clean = sparse(rows, cols, vals, m, n);
    good_users = setdiff([1:n], bad_users);
    train_clean = train_clean[:,good_users];
    #train_clean = dropzeros(train_clean);

    (m, n) = size(train_clean);
    println("after cleaning, number of users: ", n, ", number of items: ", m);
    rows = rowvals(train_clean);
    vals = nonzeros(train_clean);
    cols = zeros(nnz(train_clean));
    cc = 0;
    for i = 1:n
        rg = nzrange(train_clean, i);
        idx = rows[rg];
        len = length(idx);
        for j = 1:len
            cc += 1;
            cols[cc] = i;
        end
    end

    writedlm( string(path, "/ratins_clean.dat"), [cols rows vals], ',' );


end
