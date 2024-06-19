import numpy as np
import ctypes
from scipy.optimize import minimize
import os
import sys
#sys.path.append('/Users/marcsalinas/Desktop/GIT_REPOS/RBM')
sys.path.append('/Users/marcsalinas/Desktop/RBM')
import functions as func

current_directory = os.getcwd()
print("Current Working directory:", current_directory)
# Define some constants
############################################################################

# list of nuclei atomic masses and numbers (to be used in calibration)
A=[16,40,48,68,90,100,116,132,144,208]
Z=[8,20,20,28,40,50,50,50,62,82]

# define nstates for neutron and proton
nstates_n = [3,6,7,10,11,11,14,16,16,22]
nstates_p = [3,6,6,7,10,11,11,11,13,16]

# define the directories to retrieve the galerkin equations and basis numbers
dirs = []
for i in range(10):
    dir = f"{A[i]},{Z[i]}/{A[i]},{Z[i]},Data"
    dirs.append(dir)

# get number of basis states for each nuclei
num_basis_states_f_list = []
num_basis_states_g_list = []
num_basis_states_c_list = []
num_basis_states_d_list = []
num_basis_meson_list = []
for i in range(10):
    num_basis_states_f, num_basis_states_g, num_basis_states_c, num_basis_states_d, num_basis_meson = func.import_basis_numbers(A[i],Z[i])
    f_basis, g_basis, c_basis, d_basis, S_basis, V_basis, B_basis, A_basis = func.get_basis(A[i],Z[i],nstates_n[i],nstates_p[i])
    num_basis_states_f_list.append(num_basis_states_f)
    num_basis_states_g_list.append(num_basis_states_g)
    num_basis_states_c_list.append(num_basis_states_c)
    num_basis_states_d_list.append(num_basis_states_d)
    num_basis_meson_list.append(num_basis_meson)

# import exp data
exp_data_full = func.load_data("exp_data.txt")
exp_data = exp_data_full[:,2:]

# initial energy guesses
energy_guess_p_list = []
energy_guess_n_list = []
for i in range(10):
    energy_guess_p = [50.0 for j in range(nstates_p[i])]
    energy_guess_n = [50.0 for j in range(nstates_n[i])]
    energy_guess_n_list.append(energy_guess_n)
    energy_guess_p_list.append(energy_guess_p)

# Initialization for c libraries
############################################################################
libraries = []

# Load libraries
for i in range(10):
    lib = ctypes.CDLL(f'./{A[i]},{Z[i]}/c_functions.so')
    libraries.append(lib)

# Set up argument types for all libraries
for lib in libraries:
    lib.c_function.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                               np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                               np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
    lib.c_function.restype = None
    lib.compute_jacobian.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                      np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                      np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
    lib.compute_jacobian.restype = None
    lib.BA_function.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'), 
                                np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
    lib.BA_function.restype = ctypes.c_double

    lib.Wkskin.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
    lib.Wkskin.restype = ctypes.c_double

    lib.Rch.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
    lib.Rch.restype = ctypes.c_double


def c_function_wrapper(lib):
    def wrapper(x, params):
        y = np.empty_like(x, dtype=np.double)
        lib.c_function(x, y, params)
        return y
    return wrapper

def jacobian_wrapper(lib):
    def wrapper(x, params):
        jac = np.empty((len(x), len(x)), dtype=np.double)
        lib.compute_jacobian(x, jac.reshape(-1), params)
        return jac.T
    return wrapper

def BA_wrapper(lib):
    def wrapper(x,params):
        BA = lib.BA_function(x,params)
        return BA
    return wrapper

def Wkskin_wrapper(lib):
    def wrapper(x):
        res = lib.Wkskin(x)
        return res
    return wrapper

def Rch_wrapper(lib):
    def wrapper(x):
        res = lib.Rch(x)
        return res
    return wrapper

# Create wrapper functions for all libraries
wrapper_functions_list = []
jacobian_wrappers_list = []
BA_wrappers_list = []
Wkskin_wrappers_list = []
Rch_wrappers_list = []
for lib in libraries:
    wrapper_functions_list.append(c_function_wrapper(lib))
    jacobian_wrappers_list.append(jacobian_wrapper(lib))
    BA_wrappers_list.append(BA_wrapper(lib))
    Wkskin_wrappers_list.append(Wkskin_wrapper(lib))
    Rch_wrappers_list.append(Rch_wrapper(lib))

# Import the raw data and format it for use in the RBM code
########################################################################

MC28_DAT = np.loadtxt("LIGO_paper/MCMC_complete_28.txt")
full_couplings = MC28_DAT[:,7:17] # gsoms2, gwomw2, gpomp2, gdomd2, kappa, lambda, zeta, xi, lambda_v, lambda_s
frmt_couplings = full_couplings[:,[0,1,2,4,5,6,8]]

# need to add a new first column with the scalar mass (ms, gsoms2, gwomw2, gpomp2, kappa, lambda, zeta, lambda_v)
ms_column = np.full((frmt_couplings.shape[0], 1), 500) # initial guess for the scalar mass of 500
couplings = np.append(ms_column, frmt_couplings, axis=1)

# multiply the other couplings by their respective masses
mw = 782.5
mp = 763.0
for i in range(len(couplings)):
    couplings[i,1] = couplings[i,1]*(couplings[i,0]**2)
    couplings[i,2] = couplings[i,2]*(mw**2)
    couplings[i,3] = couplings[i,3]*(mp**2)

####################################################################
# chi-square for single nucleus
def chisq(exp_data,BA_mev_th, Rch_th, FchFwk_th):
    res = (exp_data[0]-BA_mev_th)**2/exp_data[1]**2
    if (exp_data[2] != -1):
        res = res + (exp_data[2]-Rch_th)**2/exp_data[3]**2
    return res

# construct chi-square function to be minimized for a given scalar mass (ms)
def total_chisq(ms, n_energies_list, p_energies_list, params, printres):
    res = 0.0
    params[1] = params[1]/(params[0]**2)*ms[0]**2
    params[0] = ms[0]
    for i in range(10):
        BA_mev_th, Rch_th, Fch_Fwk_th, en_n, en_p = func.hartree_RBM(A[i],nstates_n[i],nstates_p[i],num_basis_states_f_list[i],num_basis_states_g_list[i],num_basis_states_c_list[i],num_basis_states_d_list[i],num_basis_meson_list[i],params,wrapper_functions_list[i],BA_wrappers_list[i],Rch_wrappers_list[i],Wkskin_wrappers_list[i],n_energies_list[i],p_energies_list[i],jac=jacobian_wrappers_list[i])
        n_energies_list[i] = en_n
        p_energies_list[i] = en_p
        if (printres == True):
            print(A[i],BA_mev_th,Rch_th)
        res = res + chisq(exp_data[i,:],BA_mev_th,Rch_th,Fch_Fwk_th)
        if (abs(BA_mev_th) > 9.0 or abs(BA_mev_th) < 7.0):
            print("error: ",params)
    return res

'''
ms0 = [501.5] # initial guess for ms
with open("MCMC28_finite.txt",'w') as output_file:
    for i in range(len(couplings)):
        params = np.array(couplings[i,:])
        result = minimize(total_chisq,x0=ms0,args=(energy_guess_n_list,energy_guess_p_list,params,False,),method='Powell',tol=0.8)
        print(i,result.x[0],result.nfev)
        print(result.x[0],end='  ',file=output_file)
        for j in range(1,7):
            print(couplings[i,j],end= '  ',file=output_file)
        print(couplings[i,7],file=output_file)
        res = total_chisq(result.x,energy_guess_n_list,energy_guess_p_list,params,True)
        print(res)
'''

mean_params = [0.0004075061752, 0.000270793293, 0.0006472945187, 4.231869706, -0.01817572428, 0.000572921399725, 0.04265150996]
params = np.append([500.0], mean_params)
params[1] = params[1]*params[0]**2
params[2] = params[2]*782.5**2
params[3] = params[3]*763.0**2
print(params)
result = minimize(total_chisq,x0=[501.5],args=(energy_guess_n_list,energy_guess_p_list,params,False,),method='Powell')
print(params)
res = total_chisq(result.x,energy_guess_n_list,energy_guess_p_list,params,True)
print(np.exp(-0.5*res))
