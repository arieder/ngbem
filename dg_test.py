import ngsolve as ngs

from netgen.csg import *
from ngsolve import *


geo1 = CSGeometry()
#geo1.Add (OrthoBrick(Pnt(-0.5,-0.5,-0.5),Pnt(0.5,0.5,0.5)))
geo1.Add(Sphere(Pnt(0,0,0),1))



order=2
print("order= ",order)
m1 = geo1.GenerateMesh (maxh=0.05)
mesh = ngs.Mesh(m1)


cf=ngs.sqrt(ngs.x**2+ngs.y**2+ngs.z**2)#ngs.sin(ngs.x)


L = L2(mesh, order=order)
uv=GridFunction(L)
uv.Set(cf)

import bempp.api
import ngbem


bem_dc,trace_matrix,normal_trace_matrix  = ngbem.L2_trace(L,also_give_dn=True)
print("building traces complete")

def real_trace(x, n, domain_index, result):
    import numpy as np;    
    result[:] = sqrt(x[0]**2+x[1]**2+x[2]**2)

def test_trace(x):
    import numpy as np;    
    return sqrt(x[0]**2+x[1]**2+x[2]**2)


def test_neumann_trace(x):
    import numpy as np;    
    return 1


tr_uex = bempp.api.GridFunction(bem_dc,
                                fun=real_trace )

print(trace_matrix.shape, "vs ",uv.vec.FV().NumPy().shape)
coefs=trace_matrix.dot(uv.vec.FV().NumPy());
tr_u=bempp.api.GridFunction(bem_dc,coefficients=coefs);
coefs2=normal_trace_matrix.dot(uv.vec.FV().NumPy());
tr_dnu=bempp.api.GridFunction(bem_dc,coefficients=coefs);


#tr_u.plot();
tr_dnu.plot();

gfe=tr_u-tr_uex;
print("error: ",tr_u.relative_error(test_trace))
print("neumann error: ",tr_dnu.relative_error(test_neumann_trace))
