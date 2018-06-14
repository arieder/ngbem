
#from mpi4py import MPI
import ngsolve as ngs
from netgen.csg import *
from netgen.meshing import Point3d
cube = OrthoBrick( Pnt(-1,-1,-1), Pnt(1,1,1) )

geo = CSGeometry()
geo.Add (cube)

from  ngsolve import *;

mesh = Mesh(geo.GenerateMesh(maxh=0.4))

Draw(mesh);


V=HCurl(mesh,order=0,dirichlet=[0,1,2,3,4,5,6,7,8],type1=True);

uex=(cos(x),sin(x),z);

def cross(a, b):
    c = (a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0])
    return CoefficientFunction(c);

normal=ngs.specialcf.normal(3);
#factor=sqrt(normal2[0]*normal2[0]+normal2[1]*normal2[1]+normal2[2]*normal2[2])
#normal=(1/factor)*normal2



u=ngs.GridFunction(V);
u.Set(CoefficientFunction(uex),ngs.VOL);

Draw(u,mesh,"u")


import bempp.api
import ngbem
import maxwell_ngbem as ngbem_rt0;
import numpy as np;
print("trace")
bem_c,trace_matrix  = ngbem_rt0.HCurl_trace(V)
print("done trace")


def incident_field(x):
    from math import sin,cos
    import numpy as np;
    return np.array([cos(x[0]),sin(x[0]),x[2]]);

def tangential_trace(x, n, domain_index, result):
    import numpy as np;
    n/=np.linalg.norm(n,2);
    result[:] = np.cross(incident_field(x), n, axis=0)

tr_uex = bempp.api.GridFunction(bem_c,
                                fun=tangential_trace )



coefs=trace_matrix.dot(u.vec.FV().NumPy());
tr_u=bempp.api.GridFunction(bem_c,coefficients=coefs);
tr_u.plot();
tr_uex.plot();
