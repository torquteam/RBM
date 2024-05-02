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
