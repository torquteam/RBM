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