# to compile cython code run
python setup.py build_ext --inplace

# to compile c files
gcc -shared -o c_functions.so -fPIC c_functions.c -O3
gcc -shared -o c_functions_greedy.so -fPIC c_functions_greedy.c -O3

error quantification and speed check (BA (mev), Rch (fm), Fch-Fwk)
16,8 errors: 0.005591133270094026 BA   0.00024253602835381472 Rch
    286.72s  0.1057s  0.0021s/run 2700x
             0.0617s  0.0012s/run 4700x with average energy guess

40,20 errors: 0.003142295097353589 BA   0.0003143557466006275 Rch
    425.15s  0.2894s  0.0058s/run  1400x

48,20 errors: 0.0034760367763638556 BA   0.000984690460275557 Rch   9.381093416981041e-05 Wkskin
    637.77s  0.3836s  0.0077s/run   1662x

68,28 errors: 0.013286837635482414 BA    0.002364312797090937 Rch
    669.3s  0.3958s  0.0079s/run   1691x

90,40 errors: 0.003194898683670431 BA   0.0013598831524445209 Rch
    911.18s  0.7424s  0.0148s/run   1227x

100,50 errors: 0.005056040779349331 BA   0.0009252395832253057 Rch
    771.28s  1.0711s  0.0214s/run   720x

116,50 errors: 0.006672802377711733 BA   0.0009907327721998982 Rch
    754.65s  1.1467s  0.0229s/run   658x

132,50 erros: 0.00723245443963691 BA   0.0015607256708610074 Rch
    898.4s  1.3536s  0.0271s/run   663x

144,62 errors: 0.006274882133820441 BA   0.001452542506991694 Rch
    1272.1s  1.7212s  0.0344s/run   739x

208,82 errors: 0.01853566916033575 BA   0.0010335180766631068 Rch   0.0003249672772689055 FchFwk
    1309.4s  2.4966s  0.0499s/run   524x

# This works for 208Pb but error is large
3  2  3  2  2  3  2  2  3  5  2  2  3  2  3  3  2  3  2  3  4  3
3  2  3  2  2  3  2  2  3  5  2  2  3  2  3  3  2  3  2  3  4  3
3  2  4  2  3  4  2  2  3  5  2  2  3  2  5  3
3  2  4  2  3  4  2  2  3  5  2  2  3  2  5  3
7
7
7
7

1201 point grid
R_fm = convergence_help*1.1*pow(A,1.0/3.0);
error = 0.0396 BA

# This works with lower error
3  2  3  2  2  3  2  2  3  5  2  2  3  2  3  3  2  3  2  3  4  3
3  2  3  2  2  3  2  2  3  5  2  2  3  2  3  3  2  3  2  3  4  3
3  2  4  2  3  4  2  2  3  5  2  2  3  2  5  3
3  2  4  2  3  4  2  2  3  5  2  2  3  2  5  3
7
7
7
6

801 point grid
R_fm = convergence_help*1.1*pow(A,1.0/3.0);
error = 0.018 BA

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

-18.3954  1  2  1.5  -2  1  2d3;2
-18.3414  0  5  5.5  6  1  1h11;2

from greedy
4  3  3  2  2  3  2  2  3  5  2  2  3  3  2  3  2  3  2  3  4  3
4  3  3  2  2  3  2  2  3  5  2  2  3  3  2  3  2  3  2  3  4  3
4  3  4  2  4  4  2  2  3  5  2  2  3  2  5  3
4  3  4  2  4  4  2  2  3  5  2  2  3  2  5  3
7
7
7
7