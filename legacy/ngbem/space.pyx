from cython.operator cimport dereference as deref
from bempp.core.space cimport Space
from libcpp.vector cimport vector
from bempp.core.utils.shared_ptr cimport reverse_const_pointer_cast
from bempp.core.utils.shared_ptr cimport const_pointer_cast

#from cython.operator cimport dereference as dereffrom bempp.core.assembly.discrete_boundary_operator cimport RealDiscreteBoundaryOperator, ComplexDiscreteBoundaryOperator,c_DiscreteBoundaryOperator


import bempp.api.space;
import numpy as np;

    

def c_trace_space(Grid grid,kind, fespace):
    cdef Space s = Space()
    cdef int dof_mode = 0


    if kind=="P":
        s.impl_.assign(reverse_const_pointer_cast(
            shared_ptr[c_Space[double]](ng_trace_space[double](grid.impl_, <PyObject *>fespace))))
        
    elif kind=="DP":        
        return bempp.core.space.function_space(grid,kind,fespace.globalorder-1);

    print("returning");
    return s

def bempp_to_ng_map(Space space):
    cdef shared_ptr[const c_Space[double]] c_space;
    c_space = space.impl_;
    return c_bempp_to_ng_map(c_space);   


class TraceSpace(bempp.api.space.Space):
    def __init__(self,grid, fespace,comp_key=None):
        from bempp.core.space.space import function_space as _function_space
        from bempp.api.assembly.functors import scalar_function_value_functor

        domains=None
        closed=None
        strictly_on_segment=False
        
        super(TraceSpace, self).__init__(
           c_trace_space(grid._impl, "P",fespace), comp_key)
        


        self._order = fespace.globalorder;
        self._has_non_barycentric_space = True
        self._non_barycentric_space = self
        if not closed:
            self._discontinuous_space = bempp.api.space.function_space(grid, "DP", self._order, domains=domains, closed=closed, reference_point_on_segment=False, element_on_segment=True)
        else:
            self._discontinuous_space = bempp.api.space.function_space(grid, "DP", self._order, domains=domains, closed=closed, reference_point_on_segment=True, element_on_segment=strictly_on_segment)
            
        self._super_space = self._discontinuous_space
        self._evaluation_functor = scalar_function_value_functor()
        self._is_barycentric = False
        self._grid = grid

        self.fespace=fespace;

        self.isNg=True;
        print("made a TraceSpace")
    def bempp_to_ng_map(self):
        return bempp_to_ng_map(self._impl);
                                                                                                                    
def trace_space(grid,kind,fespace):
    if kind=="P":
        return TraceSpace(grid,fespace)
    else:
        raise NotImplementedError("only traces of piecwiese continuous polynomial spaces are supported for now")        
    




