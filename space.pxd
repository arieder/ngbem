from bempp.core.utils cimport shared_ptr, complex_double
from bempp.core.grid.grid cimport Grid, c_Grid
from libcpp cimport complex as ccomplex, bool as cbool
from libcpp.string cimport string
from libcpp.vector cimport vector
from cpython cimport *
from bempp.core.space cimport Space,c_Space
from bempp.core.assembly.discrete_boundary_operator cimport c_DiscreteBoundaryOperator

import numpy as np
cimport numpy as np

cdef extern from "solve.hpp":
    cdef cppclass c_FESpace "ngsolve::FESpace"
        


# Define all possible spaces
cdef extern from "pyngbem.cpp" namespace "NgBem":
    cdef shared_ptr[c_Space[T]] ng_trace_space[T](const shared_ptr[c_Grid]& grid, PyObject* object)    
    cdef np.ndarray[np.int32_t, ndim=1] c_bempp_to_ng_map(const shared_ptr[const c_Space[double] ]& space );

