pragma circom 2.1.9;

template Multiplier() {
    signal input x;
    signal input y;
    signal output z;

    z <== x * y;
}

component main = Multiplier();
