# Cythonized functions
# Create a new file with a .pyx extension, e.g., mymodule.pyx

# mymodule.pyx
import numpy as np
cimport numpy as np
import math
from libc.math cimport M_PI

cdef double conv_r0_en = 0.08405835143769969
cdef double enscale_mev = 13.269584506383948

def compute_fields(np.ndarray[np.float64_t, ndim=2] f_coeff,
                   np.ndarray[np.float64_t, ndim=2] g_coeff,
                   np.ndarray[np.float64_t, ndim=2] c_coeff,
                   np.ndarray[np.float64_t, ndim=2] d_coeff,
                   int nstates_n, int nstates_p,
                   np.ndarray[np.float64_t, ndim=3] f_basis,
                   np.ndarray[np.float64_t, ndim=3] g_basis,
                   np.ndarray[np.float64_t, ndim=3] c_basis,
                   np.ndarray[np.float64_t, ndim=3] d_basis):

    cdef np.ndarray[np.float64_t, ndim=2] f_fields_approx
    cdef np.ndarray[np.float64_t, ndim=2] g_fields_approx
    cdef np.ndarray[np.float64_t, ndim=2] c_fields_approx
    cdef np.ndarray[np.float64_t, ndim=2] d_fields_approx

    f_fields_approx = np.transpose([np.dot(f_basis[i], f_coeff[i]) for i in range(nstates_n)])
    g_fields_approx = np.transpose([np.dot(g_basis[i], g_coeff[i]) for i in range(nstates_n)])
    c_fields_approx = np.transpose([np.dot(c_basis[i], c_coeff[i]) for i in range(nstates_p)])
    d_fields_approx = np.transpose([np.dot(d_basis[i], d_coeff[i]) for i in range(nstates_p)])

    return f_fields_approx, g_fields_approx, c_fields_approx, d_fields_approx

def compute_meson_fields(np.ndarray[np.float64_t, ndim=1] s_coeff,
                         np.ndarray[np.float64_t, ndim=1] v_coeff,
                         np.ndarray[np.float64_t, ndim=1] b_coeff,
                         np.ndarray[np.float64_t, ndim=1] a_coeff,
                         np.ndarray[np.float64_t, ndim=2] S_basis,
                         np.ndarray[np.float64_t, ndim=2] V_basis,
                         np.ndarray[np.float64_t, ndim=2] B_basis,
                         np.ndarray[np.float64_t, ndim=2] A_basis):

    cdef np.ndarray[np.float64_t, ndim=1] s_field_approx
    cdef np.ndarray[np.float64_t, ndim=1] v_field_approx
    cdef np.ndarray[np.float64_t, ndim=1] b_field_approx
    cdef np.ndarray[np.float64_t, ndim=1] a_field_approx

    s_field_approx = np.dot(S_basis, s_coeff)
    v_field_approx = np.dot(V_basis, v_coeff)
    b_field_approx = np.dot(B_basis, b_coeff)
    a_field_approx = np.dot(A_basis, a_coeff)

    return s_field_approx, v_field_approx, b_field_approx, a_field_approx

def get_densities(np.ndarray[np.double_t, ndim=1] r_vec,
                         np.ndarray[np.double_t, ndim=2] f_fields_approx,
                         np.ndarray[np.double_t, ndim=2] g_fields_approx,
                         np.ndarray[np.double_t, ndim=2] c_fields_approx,
                         np.ndarray[np.double_t, ndim=2] d_fields_approx,
                         np.ndarray[np.double_t, ndim=2] state_info_n,
                         np.ndarray[np.double_t, ndim=2] state_info_p):

    cdef np.ndarray[np.double_t, ndim=1] sdensn = np.zeros_like(r_vec)
    cdef np.ndarray[np.double_t, ndim=1] vdensn = np.zeros_like(r_vec)
    cdef np.ndarray[np.double_t, ndim=1] tdensn = np.zeros_like(r_vec)
    cdef np.ndarray[np.double_t, ndim=1] sdensp = np.zeros_like(r_vec)
    cdef np.ndarray[np.double_t, ndim=1] vdensp = np.zeros_like(r_vec)
    cdef np.ndarray[np.double_t, ndim=1] tdensp = np.zeros_like(r_vec)
    cdef Py_ssize_t i, j

    cdef Py_ssize_t nstates_n = state_info_n.shape[0]
    cdef Py_ssize_t nstates_p = state_info_p.shape[0]

    for i in range(nstates_n):
        for j in range(r_vec.shape[0]):
            sdensn[j] = sdensn[j] + state_info_n[i,2]*(2*state_info_n[i,0]+1)/(4*M_PI*r_vec[j]**2) * (f_fields_approx[j,i]**2 - g_fields_approx[j,i]**2)
            vdensn[j] = vdensn[j] + state_info_n[i,2]*(2*state_info_n[i,0]+1)/(4*M_PI*r_vec[j]**2) * (f_fields_approx[j,i]**2 + g_fields_approx[j,i]**2)
            tdensn[j] = tdensn[j] + 2.0*state_info_n[i,2]*(2*state_info_n[i,0]+1)/(4*M_PI*r_vec[j]**2) * (f_fields_approx[j,i] * g_fields_approx[j,i])

    for i in range(nstates_p):
        for j in range(r_vec.shape[0]):
            sdensp[j] = sdensp[j] + state_info_p[i,2]*(2*state_info_p[i,0]+1)/(4*M_PI*r_vec[j]**2) * (c_fields_approx[j,i]**2 - d_fields_approx[j,i]**2)
            vdensp[j] = vdensp[j] + state_info_p[i,2]*(2*state_info_p[i,0]+1)/(4*M_PI*r_vec[j]**2) * (c_fields_approx[j,i]**2 + d_fields_approx[j,i]**2)
            tdensp[j] = tdensp[j] + 2.0*state_info_p[i,2]*(2*state_info_p[i,0]+1)/(4*M_PI*r_vec[j]**2) * (c_fields_approx[j,i] * d_fields_approx[j,i])

    return sdensn, vdensn, tdensn, sdensp, vdensp, tdensp