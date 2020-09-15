#from mpi4py import MPI
from netgen.csg import *


cube = OrthoBrick( Pnt(-1,-1,-1), Pnt(1,1,1) )

geo = CSGeometry()
geo.Add (cube)

from  ngsolve import *;

mesh = Mesh(geo.GenerateMesh(maxh=1))

Draw(mesh);

import ngbem;
order=1;
V=H1(mesh,order=order)
u = GridFunction (V)

##set up the FEM parts
def cosh(t):
    return 0.5*(exp(t)+exp(-t));
# the right hand side
f = LinearForm (V)

[ut,vt]=V.TnT()

# the bilinear-form
a = BilinearForm (V)
a += grad(ut)*grad(vt)*dx


###a bilinear form which is only used for preconditioning
b= BilinearForm(V);
b += (grad(ut)*grad(vt)+ut*vt)*dx


c = Preconditioner(b, type="multigrid");

a.Assemble()
b.Assemble()
#c.Update();
#refine multiple times
for j in range(0,3):
    mesh.Refine()
    V.Update();
    a.Assemble();
    b.Assemble();
    #c.Update();
    u.Update();
    f.Assemble();


Draw(u);
import bempp.api;
import numpy as np;
import scipy;


import bempp.core
bempp.core.opencl_kernels.show_available_platforms_and_devices()
bempp.core.opencl_kernels.default_device()

bempp.api.DEFAULT_DEVICE_INTERFACE="opencl"
print("device: ",bempp.api.DEFAULT_DEVICE_INTERFACE)


#bempp.api.global_parameters.hmat.eps=1E-09;
#bempp.api.global_parameters.hmat.max_rank=4096;


#increase the quadrature order. Otherwise higher order does not work

p_inc=order+2;
#bempp.api.global_parameters.quadrature.double_singular += p_inc
#bempp.api.global_parameters.quadrature.near.double_order += p_inc
#bempp.api.global_parameters.quadrature.medium.double_order += p_inc
#bempp.api.global_parameters.quadrature.far.double_order += p_inc
#bempp.api.global_parameters.quadrature.near.single_order += p_inc
#bempp.api.global_parameters.quadrature.medium.single_order += p_inc
#bempp.api.global_parameters.quadrature.far.single_order +=  p_inc


#set up the BEM spaces
[bem_c,trace_matrix]=ngbem.H1_trace(V);
bem_dc=bempp.api.function_space(bem_c.grid,'DP',order-1);


##the exact solution is exp(x)*sin(y) on the interior and 1/r on the exterior domain
@bempp.api.real_callable
def dirichlet_fun(x,n,domain_index,result):
    #from math import cosh,sinh;
    r=np.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
    result[0]=np.exp(x[0])*np.sin(x[1])-1.0/r;

@bempp.api.real_callable
def neumann_fun(x,n,domain_index,result):
    #import math;
    r=np.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
    result[0]=np.exp(x[0])*np.sin(x[1])*n[0] + np.exp(x[0])*np.cos(x[1])*n[1]+( 1/( r*r*r ) )*( np.dot(x,n) );





print("assembling the boundary operators")

from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
from bempp.api.assembly.discrete_boundary_operator import InverseSparseDiscreteBoundaryOperator
##set up the bem
bempp.api.VECTORIZATION_MODE = "novec" 
sl=bempp.api.operators.boundary.laplace.single_layer(bem_dc,bem_c,bem_dc,assembler="fmm", device_interface="opencl")
dl=bempp.api.operators.boundary.laplace.double_layer(bem_c,bem_c,bem_dc)#,assembler="fmm", device_interface="opencl")
id_op=bempp.api.operators.boundary.sparse.identity(bem_dc,bem_dc,bem_c)
id_op2=bempp.api.operators.boundary.sparse.identity(bem_c,bem_c,bem_dc)


block=np.ndarray([2,2],dtype=np.object);
block[0,0]=ngbem.NgOperator(a);
block[0,1]=-trace_matrix.T * id_op.weak_form().A;

from scipy.sparse.linalg.interface import LinearOperator

trace_op = LinearOperator(trace_matrix.shape, lambda x:trace_matrix*x)
rhs_op1=0.5*id_op2 - dl;
block[1,0]=rhs_op1.weak_form()*trace_op;
block[1,1]=sl.weak_form();
blockOp=BlockedDiscreteOperator(block);



#set up a block-diagonal preconditioner
p_block=np.ndarray([2,2],dtype=np.object);
p_block[0,0]=ngbem.NgOperator(c,a);
p_block[1,1]= InverseSparseDiscreteBoundaryOperator(
        bempp.api.operators.boundary.sparse.identity(bem_dc, bem_dc, bem_dc).weak_form())#np.identity(bem_dc.global_dof_count)

p_blockOp=BlockedDiscreteOperator(p_block);


F=np.zeros(blockOp.shape[1],blockOp.dtype);

F[0:V.ndof]=f.vec;



dirichlet_data=bempp.api.GridFunction(bem_c,fun=dirichlet_fun);
neumann_data=bempp.api.GridFunction(bem_dc,dual_space=bem_c,fun=neumann_fun);

g=rhs_op1*dirichlet_data


F[V.ndof:]=g.projections(bem_dc);


F[0:V.ndof]+=trace_matrix.T * neumann_data.projections(bem_c);


# Create a callback function to count the number of iterations
it_count = 0
def count_iterations(x):
    global it_count
    it_count += 1

print("solving..:")
ux,info = scipy.sparse.linalg.gmres(blockOp, F,tol=1E-12,restart=2000, callback=count_iterations, M=p_blockOp);

print("solving took", it_count, "iterations")


# the solution field
u = GridFunction (V)
u.vec.FV().NumPy()[:]=ux[0:V.ndof];



uex=exp(x)*sin(y);


disp=GridFunction(V,"uex")
disp.Set(uex)

# print

Draw (u)
Draw(disp)
Redraw();

intOrder=max(2*order+2,5);
print ("err-u:   ", sqrt(Integrate( (uex-u)*(uex-u),mesh,order=intOrder))) #/sqrt(Integrate(uex*uex,mesh,order=intOrder)))
