import numpy as np
import ctypes
import functions as func
import time
import bulk2params as trans

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
        return jac
    return wrapper

# Create wrapper functions for all libraries
wrapper_functions_list = []
jacobian_wrappers_list = []
for lib in libraries:
    wrapper_functions_list.append(c_function_wrapper(lib))
    jacobian_wrappers_list.append(jacobian_wrapper(lib))

# define nstates for neutron and proton
nstates_n = [3,6,7,10,11,11,14,16,16,22]
nstates_p = [3,6,6,7,10,11,11,11,13,16]

# define the directories
dirs = []
for i in range(10):
    dir = f"{A[i]},{Z[i]}/{A[i]},{Z[i]},Data"
    dirs.append(dir)

# specify grids
r_vecs = []
for i in range(10):
    r_vec = func.load_data(dirs[i] + "/rvec.txt")[:,0]
    r_vecs.append(r_vec)

# get basis states
num_basis_states_f_list = []
num_basis_states_g_list = []
num_basis_states_c_list = []
num_basis_states_d_list = []
num_basis_meson_list = []
f_basis_list = []
g_basis_list = []
c_basis_list = []
d_basis_list = []
S_basis_list = []
V_basis_list = []
B_basis_list = []
A_basis_list = []
for i in range(10):
    num_basis_states_f, num_basis_states_g, num_basis_states_c, num_basis_states_d, num_basis_meson = func.import_basis_numbers(A[i],Z[i])
    f_basis, g_basis, c_basis, d_basis, S_basis, V_basis, B_basis, A_basis = func.get_basis(A[i],Z[i],nstates_n[i],nstates_p[i])
    num_basis_states_f_list.append(num_basis_states_f)
    num_basis_states_g_list.append(num_basis_states_g)
    num_basis_states_c_list.append(num_basis_states_c)
    num_basis_states_d_list.append(num_basis_states_d)
    num_basis_meson_list.append(num_basis_meson)
    f_basis_list.append(f_basis)
    g_basis_list.append(g_basis)
    c_basis_list.append(c_basis)
    d_basis_list.append(d_basis)
    S_basis_list.append(S_basis)
    V_basis_list.append(V_basis)
    B_basis_list.append(B_basis)
    A_basis_list.append(A_basis)

# import state information (j, alpha, fill_frac, filetag)
state_info_n_list = []
state_info_p_list = []
for i in range(10):
    n_labels, state_file_n = func.load_spectrum( dirs[i] + "/neutron_spectrum.txt")
    p_labels, state_file_p = func.load_spectrum(dirs[i] + "/proton_spectrum.txt")
    state_info_n = state_file_n[:,[3,4,5]]
    state_info_p = state_file_p[:,[3,4,5]]
    state_info_n_list.append(state_info_n)
    state_info_p_list.append(state_info_p)

def compute_lkl(exp_data,BA_mev_th, Rch_th, FchFwk_th,lkl):
    lkl = lkl*np.exp(-0.5*(exp_data[0]-BA_mev_th)**2/exp_data[1]**2)
    if (exp_data[2] != -1):
        lkl = lkl*np.exp(-0.5*(exp_data[2]-Rch_th)**2/exp_data[3]**2)
    if (FchFwk_th != -1):
        lkl = lkl*np.exp(-0.5*(exp_data[4]-FchFwk_th)**2/exp_data[5]**2)
    return lkl

# import the experimental data and errors
exp_data_full = func.load_data("exp_data.txt")
exp_data = exp_data_full[:,2:]

# import start file and normalize
start_data = func.load_data("MCMC_startfile.txt")
bulks_0 = start_data[:,0]
stds = start_data[:,1]
bulks_p = np.empty_like(bulks_0)

#####################################################
# MCMC metrpolis hastings
#####################################################
nburnin = 20
nruns = 0
n_params = 8
mw = 782.5
mp = 763.0
n_nuclei = 10

# initialize the starting point
lkl0 = 1.0
params = trans.get_parameters(bulks_0[0],bulks_0[1],bulks_0[2],bulks_0[3],bulks_0[4],bulks_0[5],bulks_0[6],bulks_0[7],mw,mp)
for i in range(n_nuclei):
    BA_mev_th, Rch_th, Fch_Fwk_th = func.hartree_RBM(A[i],Z[i],nstates_n[i],nstates_p[i],num_basis_states_f_list[i],num_basis_states_g_list[i],num_basis_states_c_list[i],num_basis_states_d_list[i],num_basis_meson_list[i],params,wrapper_functions_list[i],f_basis_list[i],g_basis_list[i],c_basis_list[i],d_basis_list[i],S_basis_list[i],V_basis_list[i],B_basis_list[i],A_basis_list[i],state_info_n_list[i],state_info_p_list[i],r_vecs[i],jac=None)
    lkl0 = compute_lkl(exp_data[i,:],BA_mev_th,Rch_th,Fch_Fwk_th,lkl0)
print(lkl0)

# burn in phase for MCMC
acc_counts = [0]*n_params
n_check = 50
agoal = 0.5
arate = [0]*n_params

start_time = time.time()
with open("burnin_out.txt", "w") as output_file:
    for i in range(nburnin):

        # get new proposed parameters
        for j in range(n_params):
            for k in range(n_params):
                bulks_p[k] = bulks_0[k] # copy old values
            bulks_p[j] = np.random.normal(bulks_0[j],stds[j])      # change one param
            params = trans.get_parameters(bulks_p[0],bulks_p[1],bulks_p[2],bulks_p[3],bulks_p[4],bulks_p[5],bulks_p[6],bulks_p[7],mw,mp)
        
            # compute new liklihood
            lklp = 1.0
            ##########################
            for k in range(10):
                BA_mev_th, Rch_th, Fch_Fwk_th = func.hartree_RBM(A[k],Z[k],nstates_n[k],nstates_p[k],num_basis_states_f_list[k],num_basis_states_g_list[k],num_basis_states_c_list[k],num_basis_states_d_list[k],num_basis_meson_list[k],params,wrapper_functions_list[k],f_basis_list[k],g_basis_list[k],c_basis_list[k],d_basis_list[k],S_basis_list[k],V_basis_list[k],B_basis_list[k],A_basis_list[k],state_info_n_list[k],state_info_p_list[k],r_vecs[k],jac=jacobian_wrappers_list[k])
                lklp = compute_lkl(exp_data[k,:],BA_mev_th,Rch_th,Fch_Fwk_th,lklp)
            print(lklp)
            ###########################
            # metroplis hastings step
            r = np.random.uniform(0,1)
            a = lklp/lkl0
            if (a>1):
                a=1.0
            if (r <= a):
                lkl0 = lklp
                for k in range(n_params):
                    bulks_0[k] = bulks_p[k] # accept the proposed changes
                    print(f"{bulks_0[k]}",file=output_file, end='  ')
                print("",file=output_file)
                acc_counts[j]+=1   # count how many times the changes are accepted

            # rate monitoring to adjust the width of the sampling
            if ((i+1)%n_check == 0):
                arate[j] = acc_counts[j]/n_check
                acc_counts[j] = 0
                if (arate[j] < agoal):
                    stds[j] = 0.9*stds[j]      # if acceptance rate is too low then decrease the range
                elif (arate > agoal):
                    stds[j] = 1.1*stds[j]      # if acceptance rate is too high then increase the range
        print(f"{i+1} completed")
end_time = time.time()
##############################################
# end of burn in

'''
# start MCMC runs
with open("MCMC.txt", "w") as output_file:
    for i in range(nruns):

        # get new proposed parameters
        for j in range(n_params):
            for k in range(n_params):
                bulks_p[k] = bulks_0[k] # copy old values
            bulks_p[j] = np.random.normal(bulks_0[j],stds[j])      # change one param
            params = trans.get_parameters(bulks_p[0],bulks_p[1],bulks_p[2],bulks_p[3],bulks_p[4],bulks_p[5],bulks_p[6],bulks_p[7],mw,mp)
        
            # compute new liklihood
            lklp = 1.0
            ##########################
            for k in range(6):
                BA_mev_th, Rch_th, Fch_Fwk_th = func.hartree_RBM(A[k],Z[k],nstates_n[k],nstates_p[k],num_basis_states_f_list[k],num_basis_states_g_list[k],num_basis_states_c_list[k],num_basis_states_d_list[k],num_basis_meson_list[k],params,wrapper_functions_list[k],f_basis_list[k],g_basis_list[k],c_basis_list[k],d_basis_list[k],S_basis_list[k],V_basis_list[k],B_basis_list[k],A_basis_list[k],state_info_n_list[k],state_info_p_list[k],r_vecs[k],jac=jacobian_wrappers_list[k])
                lklp = compute_lkl(exp_data[k,:],BA_mev_th,Rch_th,Fch_Fwk_th,lklp)
            BA_mev_th, Rch_th, Fch_Fwk_th = func.hartree_RBM(A[6],Z[6],nstates_n[6],nstates_p[6],num_basis_states_f_list[6],num_basis_states_g_list[6],num_basis_states_c_list[6],num_basis_states_d_list[6],num_basis_meson_list[6],params,wrapper_functions_list[6],f_basis_list[6],g_basis_list[6],c_basis_list[6],d_basis_list[6],S_basis_list[6],V_basis_list[6],B_basis_list[6],A_basis_list[6],state_info_n_list[6],state_info_p_list[6],r_vecs[6],jac=None)
            lklp = compute_lkl(exp_data[6,:],BA_mev_th,Rch_th,Fch_Fwk_th,lklp)
            BA_mev_th, Rch_th, Fch_Fwk_th = func.hartree_RBM(A[7],Z[7],nstates_n[7],nstates_p[7],num_basis_states_f_list[7],num_basis_states_g_list[7],num_basis_states_c_list[7],num_basis_states_d_list[7],num_basis_meson_list[7],params,wrapper_functions_list[7],f_basis_list[7],g_basis_list[7],c_basis_list[7],d_basis_list[7],S_basis_list[7],V_basis_list[7],B_basis_list[7],A_basis_list[7],state_info_n_list[7],state_info_p_list[7],r_vecs[7],jac=jacobian_wrappers_list[7])
            lklp = compute_lkl(exp_data[7,:],BA_mev_th,Rch_th,Fch_Fwk_th,lklp)
            BA_mev_th, Rch_th, Fch_Fwk_th = func.hartree_RBM(A[8],Z[8],nstates_n[8],nstates_p[8],num_basis_states_f_list[8],num_basis_states_g_list[8],num_basis_states_c_list[8],num_basis_states_d_list[8],num_basis_meson_list[8],params,wrapper_functions_list[8],f_basis_list[8],g_basis_list[8],c_basis_list[8],d_basis_list[8],S_basis_list[8],V_basis_list[8],B_basis_list[8],A_basis_list[8],state_info_n_list[8],state_info_p_list[8],r_vecs[8],jac=None)
            lklp = compute_lkl(exp_data[8,:],BA_mev_th,Rch_th,Fch_Fwk_th,lklp)
            BA_mev_th, Rch_th, Fch_Fwk_th = func.hartree_RBM(A[9],Z[9],nstates_n[9],nstates_p[9],num_basis_states_f_list[9],num_basis_states_g_list[9],num_basis_states_c_list[9],num_basis_states_d_list[9],num_basis_meson_list[9],params,wrapper_functions_list[9],f_basis_list[9],g_basis_list[9],c_basis_list[9],d_basis_list[9],S_basis_list[9],V_basis_list[9],B_basis_list[9],A_basis_list[9],state_info_n_list[9],state_info_p_list[9],r_vecs[9],jac=jacobian_wrappers_list[9])
            lklp = compute_lkl(exp_data[9,:],BA_mev_th,Rch_th,Fch_Fwk_th,lklp)
            print(lklp)
            ###########################
            # metroplis hastings step
            r = np.random.uniform(0,1)
            a = lklp/lkl0
            if (a>1):
                a=1.0
            if (r <= a):
                lkl0 = lklp
                for k in range(n_params):
                    bulks_0[k] = bulks_p[k] # accept the proposed changes
                    print(f"{bulks_0[k]}",file=output_file, end='  ')
                print("",file=output_file)

        print(f"{i+1} completed")     
'''   
print("Took:{:.4f} seconds per step".format((end_time - start_time)/nburnin))