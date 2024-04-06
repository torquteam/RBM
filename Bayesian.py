import numpy as np
import ctypes
import functions as func
import time
import bulk2params as trans
import multiprocessing
import functools
import operator
import os
from sys import exit

current_directory = os.getcwd()
print("Current Working directory:", current_directory)

# import c functions for all nuclei
A=[16,40,48,68,90,100,116,132,144,208]
Z=[8,20,20,28,40,50,50,50,62,82]

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

# define nstates for neutron and proton
nstates_n = [3,6,7,10,11,11,14,16,16,22]
nstates_p = [3,6,6,7,10,11,11,11,13,16]

# define the directories
dirs = []
for i in range(10):
    dir = f"{A[i]},{Z[i]}/{A[i]},{Z[i]},Data"
    dirs.append(dir)

# get basis states
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

# define the lkl function for a single nucleus
def compute_lkl(exp_data,BA_mev_th, Rch_th, FchFwk_th):
    lkl = np.exp(-0.5*(exp_data[0]-BA_mev_th)**2/exp_data[1]**2)
    if (exp_data[2] != -1):
        lkl = lkl*np.exp(-0.5*(exp_data[2]-Rch_th)**2/exp_data[3]**2)
    if (FchFwk_th != -1):
        lkl = lkl*np.exp(-0.5*(exp_data[4]-FchFwk_th)**2/exp_data[5]**2)
    return lkl

# define the total liklihood
def compute_nuclei_v2(num_nuclei,params,flag,n_energies_list,p_energies_list):
    lkl = 1
    if (flag == True):
        print("flagged")
        return 0.0
    for i in range(num_nuclei):
        BA_mev_th, Rch_th, Fch_Fwk_th, en_n, en_p = func.hartree_RBM(A[i],Z[i],nstates_n[i],nstates_p[i],num_basis_states_f_list[i],num_basis_states_g_list[i],num_basis_states_c_list[i],num_basis_states_d_list[i],num_basis_meson_list[i],params,wrapper_functions_list[i],BA_wrappers_list[i],Rch_wrappers_list[i],Wkskin_wrappers_list[i],n_energies_list[i],p_energies_list[i],jac=jacobian_wrappers_list[i])
        n_energies_list[i] = en_n
        p_energies_list[i] = en_p
        #print(A[i],BA_mev_th)
        lkl = lkl*compute_lkl(exp_data[i,:],BA_mev_th,Rch_th,Fch_Fwk_th)
        if (abs(BA_mev_th) > 9.0 or abs(BA_mev_th) < 7.0):
            print("error: ",params)
            return 0.0

    return lkl

def metropolis(lkl0, lklp, bulks_0, bulks_p, acc_counts, output_file, index):
    # metroplis hastings step
    r = np.random.uniform(0,1)
    a = lklp/lkl0
    if (a>1):
        a=1.0
    if (r <= a):
        lkl0 = lklp
        for k in range(n_params):
            bulks_0[k] = bulks_p[k] # accept the proposed changes
        acc_counts[index]+=1   # count how many times the changes are accepted

def adaptive_width(n_check,arate,acc_counts,stds,agoal,index):
    if ((i+1)%n_check == 0):
        arate[index] = acc_counts[index]/n_check
        acc_counts[index] = 0
        if (arate[index] < agoal):
            stds[index] = 0.9*stds[index]      # if acceptance rate is too low then decrease the range
        elif (arate[index] > agoal):
            stds[index] = 1.1*stds[index]      # if acceptance rate is too high then increase the range

def param_change(n_params, bulks_0, bulks_p, stds, mw, mp, index):
    for k in range(n_params):
        bulks_p[k] = bulks_0[k] # copy old values
    bulks_p[index] = np.random.normal(bulks_0[index],stds[index])      # change one param
    params, flag = trans.get_parameters(bulks_p[0],bulks_p[1],bulks_p[2],bulks_p[3],bulks_p[4],bulks_p[5],bulks_p[6],bulks_p[7],mw,mp)
    return params, flag

def MCMC_worker(args):
    iterations_burn, iterations_run, process_id = args
    np.random.seed(process_id)
    start_time = time.time()
    with open(f"burnin_out_{process_id}.txt", "w") as output_file:
        for i in range(iterations_burn):
            # one sweep
            for j in range(n_params):
                # get new proposed parameters
                params, flag = param_change(n_params,bulks_0,bulks_p,stds,mw,mp,j)

                # get the lkl
                lklp = compute_nuclei_v2(n_nuclei,params,flag,energy_guess_n_list,energy_guess_p_list)

                # metroplis hastings step
                metropolis(lkl0,lklp,bulks_0,bulks_p,acc_counts,output_file,j)

                # rate monitoring to adjust the width of the sampling
                adaptive_width(n_check,arate,acc_counts,stds,agoal,j)

            # print MCMC sweep
            for k in range(n_params):
                print(f"{bulks_0[k]}",file=output_file, end='  ')
            print("",file=output_file)
            print(f"{i+1} completed")
    end_time = time.time()
    print("Burn in took:{:.4f} seconds".format((end_time-start_time)))
    ##############################################
    # end of burn in

    # start MCMC runs
    with open(f"MCMC_{process_id}.txt", "w") as output_file:
        for i in range(iterations_run):
            # one sweep
            for j in range(n_params):
                # get new proposed parameters
                params, flag = param_change(n_params,bulks_0,bulks_p,stds,mw,mp,j)

                # get lkl
                lklp = compute_nuclei_v2(n_nuclei,params,flag,energy_guess_n_list,energy_guess_p_list)

                # metroplis hastings step
                metropolis(lkl0,lklp,bulks_0,bulks_p,acc_counts,output_file,j)
            
            # print MCMC sweep
            for k in range(n_params):
                print(f"{bulks_0[k]}",file=output_file, end='  ')
            print("",file=output_file)
            print(f"{i+1} completed")

# import the experimental data and errors
exp_data_full = func.load_data("exp_data.txt")
exp_data = exp_data_full[:,2:]

# import state information (j, alpha, fill_frac, filetag)
state_info_n_list = []
state_info_p_list = []
energy_guess_p_list = []
energy_guess_n_list = []
for i in range(10):
    n_labels, state_file_n = func.load_spectrum( dirs[i] + "/neutron_spectrum.txt")
    p_labels, state_file_p = func.load_spectrum(dirs[i] + "/proton_spectrum.txt")
    energies_p = func.load_data(dirs[i] + "/proton/energies.txt")
    energies_n = func.load_data(dirs[i] + "/neutron/energies.txt")
    energy_guess_p = [np.mean(energies_p[:,j]) for j in range(nstates_p[i])]
    energy_guess_n = [np.mean(energies_n[:,j]) for j in range(nstates_n[i])]
    state_info_n = state_file_n[:,[3,4,5]]
    state_info_p = state_file_p[:,[3,4,5]]
    state_info_n_list.append(state_info_n)
    state_info_p_list.append(state_info_p)
    energy_guess_n_list.append(energy_guess_n)
    energy_guess_p_list.append(energy_guess_p)

# import start file
start_data = func.load_data("MCMC_startfile.txt")
bulks_0 = start_data[:,0]
stds = start_data[:,1]
bulks_p = np.empty_like(bulks_0)

#####################################################
# MCMC metrpolis hastings
#####################################################
nburnin = 10000
nruns = 50000
n_params = 8
mw = 782.5
mp = 763.0
n_nuclei = 10

# adaptive specifications for MCMC
acc_counts = [0]*n_params
n_check = 50
agoal = 0.3
arate = [0]*n_params

# initialize the starting point
params, flag = trans.get_parameters(bulks_0[0],bulks_0[1],bulks_0[2],bulks_0[3],bulks_0[4],bulks_0[5],bulks_0[6],bulks_0[7],mw,mp)
#params, flag = trans.get_parameters(-16.267735180987582, 0.1487504197034128, 0.5796102233938005, 228.8288441185011, 34.321244156393746, 75.94673230448575, 0.038694010541058296, 488.2194935337496,mw,mp)
#params = np.array([4.93933385e+02, 1.17600464e+02, 2.07292704e+02, 1.71160548e+04, 2.69606892e+00, 1.81131594e-03, 2.97681837e-02, 4.39006417e-02])
lkl0 = compute_nuclei_v2(n_nuclei,params,flag,energy_guess_n_list,energy_guess_p_list)

# parallelize the MCMC
n_processes = 4

inputs = [(nburnin, nruns, i) for i in range(n_processes)]
with  multiprocessing.Pool(processes=n_processes) as pool:
    results = pool.map(MCMC_worker, inputs)