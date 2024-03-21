from scipy.optimize import root
import numpy as np
import time
import ctypes
import sys
sys.path.append('/home/msals97/Desktop/RBM/RBM')
import functions as func

# Load C functions shared library
lib = ctypes.CDLL('./48,20/c_functions.so')

# Define the argument and return types of the functions
lib.c_function.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                                          np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                                          np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
lib.c_function.restype = None

lib.compute_jacobian.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                             np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                             np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
lib.compute_jacobian.restype = None

lib.BA_function.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'), np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
lib.BA_function.restype = ctypes.c_double

lib.Wkskin.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
lib.Wkskin.restype = ctypes.c_double

lib.Rch.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
lib.Rch.restype = ctypes.c_double

# Define a wrapper function that calls the C functions
def c_function_wrapper(x,params):
    y = np.empty_like(x, dtype=np.double)
    lib.c_function(x, y, params)
    return y

def compute_jacobian_wrapper(x,params):
    jac = np.empty((len(x), len(x)), dtype=np.double)
    lib.compute_jacobian(x, jac.reshape(-1),params)
    return jac.T

def BA_function(x,params):
    BA = lib.BA_function(x,params)
    return BA

def Wkskin(x):
    res = lib.Wkskin(x)
    return res

def Rch(x):
    res = lib.Rch(x)
    return res


# User Input
##################################################

# Specify the number of proton and neutron states
nstates_n = 7
nstates_p = 6
mNuc_mev = 939

# Specify the number of protons and neutrons (for file specification purposes)
A = 48
Z = 20

# import Data
##################################################
dir = f"{A},{Z}/{A},{Z},Data"
num_basis_states_f, num_basis_states_g, num_basis_states_c, num_basis_states_d, num_basis_meson = func.import_basis_numbers(A,Z)
param_set = func.load_data("param_sets_gold.txt")
actual_results = func.load_data(dir + f"/{A},{Z}Observables.txt")

# Initial guess setup
##################################################
initial_guess_array = func.initial_guess(nstates_n,nstates_p,num_basis_states_f,num_basis_states_g,num_basis_states_c,num_basis_states_d,num_basis_meson[0],num_basis_meson[1],num_basis_meson[2],num_basis_meson[3])

# Nonlinear solve
##############################################################################
start_time = time.time()
errBA = 0
errRch = 0
errWk = 0
nruns = 50
for i in range(nruns):
    params = param_set[i%50,:]
    #params = [496.939,110.349,187.695,192.927,3.26,-0.003551,0.0235,0.043377]
    params_array = np.array(params, dtype=np.double)

    solution = root(c_function_wrapper, x0=initial_guess_array, args=(params_array,), jac=compute_jacobian_wrapper, method='hybr',options={'col_deriv': 1, 'xtol': 1e-8})

    BA_mev = (BA_function(solution.x,params_array)*13.269584506383948 - 0.75*41.0*A**(-1.0/3.0))/A - 939
    Rcharge = Rch(solution.x)
    FchFwk = Wkskin(solution.x)
    print("Binding energy = ", BA_mev)
    print(f"Rch = {Rcharge}" )
    print(f"Fch - Fwk = {FchFwk}")

    # compute the average err of each observable
    errBA = errBA + abs(actual_results[i%50][0] - BA_mev)
    errRch = errRch + abs(actual_results[i%50][1] - Rcharge)
    errWk = errWk + abs(actual_results[i%50][2] - FchFwk)
    
end_time = time.time()
print("SVD took:{:.4f} seconds".format(end_time - start_time))
print("{:.4f}s/run".format((end_time - start_time)/nruns))

errBA = errBA/nruns
errRch = errRch/nruns
errWk = errWk/nruns
print(errBA, errRch, errWk)