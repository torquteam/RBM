from scipy.optimize import root
import numpy as np
import time
from scipy.linalg import svd
import math
from scipy.integrate import simps
import ctypes
import sys
sys.path.append('/home/msals97/Desktop/FSU_FINITE/ReducedBasisMethods')
import functions as func
from cyth_funcs import compute_fields, compute_meson_fields, get_densities

# Load C functions shared library
lib = ctypes.CDLL('./116,50/c_functions.so')

# Define the argument and return types of the functions
lib.c_function.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                                          np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                                          np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
lib.c_function.restype = None

lib.compute_jacobian.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                             np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                             np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
lib.compute_jacobian.restype = None

# Define a wrapper function that calls the C functions
def c_function_wrapper(x,params):
    y = np.empty_like(x, dtype=np.double)
    lib.c_function(x, y, params)
    return y

def compute_jacobian_wrapper(x,params):
    jac = np.empty((len(x), len(x)), dtype=np.double)
    lib.compute_jacobian(x, jac.reshape(-1),params)
    return jac

# User Input
##################################################

# Specify the number of proton and neutron states
nstates_n = 14
nstates_p = 11
mNuc_mev = 939

# Specify the number of protons and neutrons (for file specification purposes)
A = 116
Z = 50

# import Data
##################################################
dir = f"{A},{Z}/{A},{Z},Data"

# specify common grid
r_vec = func.load_data(dir + "/rvec.txt")[:,0]

num_basis_states_f, num_basis_states_g, num_basis_states_c, num_basis_states_d, num_basis_meson = func.import_basis_numbers(A,Z)
f_basis, g_basis, c_basis, d_basis, S_basis, V_basis, B_basis, A_basis = func.get_basis(A,Z,nstates_n,nstates_p)

# import state information (j, alpha, fill_frac, filetag)
n_labels, state_file_n = func.load_spectrum( dir + "/neutron_spectrum.txt")
p_labels, state_file_p = func.load_spectrum(dir + "/proton_spectrum.txt")
state_info_n = state_file_n[:,[3,4,5]]
state_info_p = state_file_p[:,[3,4,5]]
#############################################################################

# Nonlinear solve
##############################################################################
# Initial guess setup
start_time = time.time()
initial_guess_array = func.initial_guess(nstates_n,nstates_p,num_basis_states_f,num_basis_states_g,num_basis_states_c,num_basis_states_d,num_basis_meson[0],num_basis_meson[1],num_basis_meson[2],num_basis_meson[3])

# Set the couplings
param_set = func.load_data("new_param_set.txt")
actual_results = func.load_data(dir + f"/{A},{Z}Observables.txt")
errBA = 0
errRch = 0
for i in range(50):
    params = param_set[i,:]
    params_array = np.array(params, dtype=np.double)

    solution = root(c_function_wrapper, x0=initial_guess_array, args=(params_array,), jac=compute_jacobian_wrapper, method='hybr',options={'col_deriv': 0, 'xtol': 1e-10})

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

    # Compute densities
    sdensn, vdensn, tdensn, sdensp, vdensp, tdensp = get_densities(r_vec, f_fields_approx, g_fields_approx, c_fields_approx, d_fields_approx, state_info_n, state_info_p)

    # Compute binding energy
    BA_mev = func.get_BA(nstates_n,nstates_p,state_info_n,state_info_p,solution,s_field_approx,v_field_approx,b_field_approx,a_field_approx,sdensn,sdensp,vdensn,vdensp,params,r_vec,A)
    print("Binding energy:", BA_mev)

    # Compute Radii
    Rn, Rp, Rch = func.get_radii(A,Z,r_vec,vdensn,vdensp)
    RnRp = Rn - Rp
    print(f"Rn = {Rn}" )
    print(f"Rp = {Rp}" )
    print(f"Rn-Rp = {RnRp}" )
    print(f"Rch = {Rch}" )

    # compute the average err of each observable
    errBA = errBA + abs(actual_results[i][0] - BA_mev)
    errRch = errRch + abs(actual_results[i][1] - Rch)
end_time = time.time()
print("SVD took:{:.4f} seconds".format((end_time - start_time)/50))

errBA = errBA/50
errRch = errRch/50
print(errBA, errRch)