from scipy.optimize import root
import numpy as np
import time
import ctypes
import sys
sys.path.append('/home/msals97/Desktop/RBM/RBM')
import functions as func
from cyth_funcs import compute_fields, compute_meson_fields

# Load C functions shared library
lib = ctypes.CDLL('./208,82/c_functions.so')

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
nstates_n = 22
nstates_p = 16
mNuc_mev = 939

# Specify the number of protons and neutrons (for file specification purposes)
A = 208
Z = 82

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
    params = param_set[i,:]
    #params = [ 4.97643867e+02,  1.04424564e+02,  1.73678283e+02,  8.31714774e+01, 3.68769280e+00, -9.75818648e-03,  1.22552587e-02,  7.62789790e-04]
    params_array = np.array(params, dtype=np.double)

    solution = root(c_function_wrapper, x0=initial_guess_array, args=(params_array,), jac=None, method='hybr',options={'col_deriv': 1, 'xtol': 1e-8})
    #print(solution.x)
    BA_mev = (BA_function(solution.x,params_array)*13.269584506383948 - 0.75*41.0*A**(-1.0/3.0))/A - 939
    Rcharge = Rch(solution.x)
    FchFwk = Wkskin(solution.x)
    print("Binding energy = ", BA_mev)
    #print(f"Rch = {Rcharge}" )
    #print(f"Fch - Fwk = {FchFwk}")

    # compute the average err of each observable
    errBA = errBA + abs(actual_results[i%50][0] - BA_mev)
    errRch = errRch + abs(actual_results[i%50][1] - Rcharge)
    errWk = errWk + abs(actual_results[i%50][2] - FchFwk)

errBA = errBA/nruns
errRch = errRch/nruns
errWk = errWk/nruns
print(errBA, errRch, errWk)

f_basis, g_basis, c_basis, d_basis, S_basis, V_basis, B_basis, A_basis = func.get_basis(A,Z,nstates_n,nstates_p)

params = param_set[0,:]
params_array = np.array(params, dtype=np.double)
solution = root(c_function_wrapper, x0=initial_guess_array, args=(params_array,), jac=None, method='hybr',options={'col_deriv': 1, 'xtol': 1e-8},tol=1e-18)
#print(solution.x)

# Reconstruct the wave functions
f_coeff = np.array(func.pad([[solution.x[int(np.sum(num_basis_states_f[:j])) + i] for i in range(num_basis_states_f[j])] for j in range(nstates_n)]))
g_coeff = np.array(func.pad([[solution.x[sum(num_basis_states_f) + int(np.sum(num_basis_states_g[:j])) + i] for i in range(num_basis_states_g[j])] for j in range(nstates_n)]))
c_coeff = np.array(func.pad([[solution.x[sum(num_basis_states_f) + sum(num_basis_states_g) + int(np.sum(num_basis_states_c[:j])) + i] for i in range(num_basis_states_c[j])] for j in range(nstates_p)]))
d_coeff = np.array(func.pad([[solution.x[sum(num_basis_states_f) + sum(num_basis_states_g) + sum(num_basis_states_c) + int(np.sum(num_basis_states_d[:j])) + i] for i in range(num_basis_states_d[j])] for j in range(nstates_p)]))
f_fields_approx, g_fields_approx, c_fields_approx, d_fields_approx = compute_fields(f_coeff, g_coeff, c_coeff, d_coeff, nstates_n, nstates_p, f_basis, g_basis, c_basis, d_basis)

# Reconstruct the meson fields
total_wf_basis = sum(num_basis_states_f) + sum(num_basis_states_g) + sum(num_basis_states_c) + sum(num_basis_states_d)
s_coeff = np.array([solution.x[total_wf_basis + j] for j in range(num_basis_meson[0])])
v_coeff = np.array([solution.x[total_wf_basis + num_basis_meson[0] + j] for j in range(num_basis_meson[1])])
b_coeff = np.array([solution.x[total_wf_basis + num_basis_meson[0] + num_basis_meson[1] + j] for j in range(num_basis_meson[2])])
a_coeff = np.array([solution.x[total_wf_basis + num_basis_meson[0] + num_basis_meson[1] + num_basis_meson[2] + j] for j in range(num_basis_meson[3])])

s_field_approx, v_field_approx, b_field_approx, a_field_approx = compute_meson_fields(s_coeff, v_coeff, b_coeff, a_coeff, S_basis, V_basis, B_basis, A_basis)

import matplotlib.pyplot as plt
rvec = np.loadtxt("208,82/208,82,Data/rvec.txt")
actual = np.loadtxt("208,82/High_fidelity_sol/meson_fields.txt")
#meson = np.loadtxt("208,82/High_fidelity_sol/meson_fields.txt")/enscale_mev
#plt.plot(rvec[1:],g_basis[set][1:,0],ls='solid') #

plt.plot(rvec[1:],actual[1:,5]/13.269584506383948 ,ls='solid')#
plt.plot(rvec[1:],a_field_approx,ls='dashed')
    #plt.plot(rvec[1:],actual[1:,i+1],ls='solid') #
    #plt.plot(rvec[1:],g_fields_approx[:,i],ls='dashed')
plt.show()

end_time = time.time()
print("SVD took:{:.4f} seconds".format(end_time - start_time))
print("{:.4f}s/run".format((end_time - start_time)/nruns))
