# to compile cython code run
python setup.py build_ext --inplace

# to compile c files
gcc -shared -o c_functions.so -fPIC c_functions.c -O3
gcc -shared -o c_functions_greedy.so -fPIC c_functions_greedy.c -O3

# functions in C file for easy replacement.
double BA_function(double *x, double* params) {
    double meson_free, meson_interact, en_BA, BA;

    return BA;
}

double Rch(double*x) {
    double Rch, Rp2;

    return Rch;
}

double Wkskin(double*x) {
    double FchFwk;
    
    return FchFwk;
}

# Stability of the root finder for each of the nuclei. well conditoned implies that the solutions are insensitive to the initial search points
16,8    well conditioned [20,70]
40,20   well conditioned [20,60]
48,20   well conditioned [20,70]
68,28   well conditioned [20,70]
90,40   well conditioned [20,60]
100,50  well conditioned [20,60]
116,50  well conditioned [20,70]
132,50  mid conditioned  [40,60]
144,62  well conditioned [20,60]
208,82  ill conditioned  [50,60]

1098, 1142, 1181, 1606, 1607, 2150, 2332, 2333, 2508 index

stds
[1.24142543e-02 5.90060235e-04 4.07984506e-03 2.12421685e+00 4.65884299e-01 4.43354386e+00 1.39905195e-03 6.95795170e-01]

# Speed up quantification

error quantification (BA (mev), Rch (fm), Fch-Fwk)
16,8 errors: 
    5.73s/run   0.0034s/run     1685x speedup

40,20 errors:
    8.50s/run   0.0109s/run     779x speedup

48,20 errors:
    12.75s/run  0.0130s/run     980x speedup

68,28 errors:
    13.38s/run  0.0160s/run     836x speedup

90,40 errors:
    18.22s/run  0.0285s/run     639x speedup

100,50 errors:
    15.42s/run  0.0381s/run     404x speedup

116,50 errors:
    15.09s/run  0.0447s/run     342x speedup

132,50 errors:
    17.96s/run  0.0537s/run     334x speedup

144,62 errors:
    25.44s/run  0.0701s/run     363x speedup

208,82 errors:
    26.18s/run  0.237s/run      110x speedup