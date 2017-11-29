using StatsBase

function main(path)
    # Y true score, X training score
    Y = readdlm(string(path, "/ratings.dat"), ',');
    X = readdlm(string(path, "/train_positive.csv"), ',');
    #(rows, cols, vals) = findnz(X);
    rows = round(Int64, X[:,1]);
    cols = round(Int64, X[:,2]);
    n = length(rows);
    ss = sample([1:n;], Int(1e4), replace = false);

    U = round(Int64, Y[:,1]);
    I = round(Int64, Y[:,2]);
    R = Y[:,3]
    n = maximum(U); m = maximum(I);
    Y = sparse(U, I, R, n, m);

    for i in ss
        row_idx = rows[i];
        col_idx = cols[i];
        if Y[row_idx, col_idx] == 0
            println("wrong!")
            break;
        end
    end
    println("train positive complete!")

    Y = readdlm(string(path, "/train_positive.csv"), ',');
    U = round(Int64, Y[:,1]);
    I = round(Int64, Y[:,2]);
    R = Y[:,3]
    n = maximum(U); m = maximum(I);
    Y = sparse(U, I, R, n, m);

    for r = 1:5
        X = readdlm(string(path, "/negsamples", string(r), ".csv"), ',');
        rows = round(Int64, X[:,1]);
        cols = round(Int64, X[:,2]);
        n = length(rows);
        ss = sample([1:n;], Int(1e4), replace = false);
        for i in ss
            row_idx = rows[i];
            col_idx = cols[i];
            if Y[row_idx, col_idx] != 0
                println("wrong!")
                break;
            end
        end
        println("neg sampling ", r, " complete!")


    end


end
