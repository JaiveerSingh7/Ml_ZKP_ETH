pragma circom 2.1.9;

template LinearRegression(n) {
    signal input x[n];          // private inputs
    signal input weights[n];    // public weights
    signal input bias;          // public bias
    signal output y;            // public output

    signal tmp[n];
    signal acc[n + 1];

    acc[0] <== bias;

    for (var i = 0; i < n; i++) {
        tmp[i] <== x[i] * weights[i];
        acc[i + 1] <== acc[i] + tmp[i];
    }

    y <== acc[n];
}

component main = LinearRegression(2);
