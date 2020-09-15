import ngsolve as ngs
import bempp.api
import numpy as np

bempp.api.DEFAULT_DEVICE_INTERFACE="opencl"

from math import pi,sqrt
k =  1

print("k=",k)
eta= -k


nu=1
b0=0.1
d0=0.1
a0=10


use_sphere=False
from netgen.csg import *
from ngsolve import *

from ctypes import CDLL



from netgen.csg import *
geo1 = CSGeometry()
if(use_sphere):
    geo1.Add (Sphere(Pnt(0,0,0),1))
else:
    geo1.Add (OrthoBrick(Pnt(-0.5,-0.5,-0.5),Pnt(0.5,0.5,0.5)))

errors=list();
it_counts=list();

from  ngbem import *

import sys
order=1
num_refines=3

print("doing order ",order," and ",num_refines," refines")

m1 = geo1.GenerateMesh (maxh=1)
mesh = ngs.Mesh(m1)
ng_space=ngs.L2(mesh,order=order,complex=True,dgjumps=True)

u,v=ng_space.TnT();

h=ngs.specialcf.mesh_size
n=ngs.specialcf.normal(3);

h0=h#ngs.Parameter(1)
beta=b0*k*h/order
delta=d0*k*h0/order
 
def ngmax(x,y):
    return ngs.IfPos(x-y,x,y)
alpha=a0*order*order/(k*ngmax(h,h.Other()))

jump_un = n*(u-u.Other())
jump_vn = n*(v-v.Other())

jump_dudn=n*(grad(u)-grad(u.Other()))
jump_dvdn=n*(grad(v)-grad(v.Other()))
    
mean_u=0.5*(u+u.Other())
mean_v=0.5*(v+v.Other())
mean_du = 0.5 * (grad(u)+grad(u.Other()))
mean_dv = 0.5 * (grad(v)+grad(v.Other()))

diff_coef=1

blfA=ngs.BilinearForm(ng_space)

ikinv=1/(1j*k);

blfA_Ak = nu*grad(u)*grad(v) * dx \
          - (k*diff_coef)**2*u*v*dx \
          - nu*(jump_un*mean_dv)*dx(skeleton=True) \
          - nu*(mean_du*jump_vn)*dx(skeleton=True) \
          - ikinv*beta*nu*jump_dudn*jump_dvdn*dx(skeleton=True) \
          + 1j*k*alpha*jump_un*jump_vn*dx(skeleton=True)

blfA_Bk = - delta*ikinv*(grad(u)*n)*(grad(v)*n)*ds(skeleton=True,definedon=~mesh.Boundaries('x')) \
                                                            - delta*u*(grad(v)*n)*ds(skeleton=True,definedon=~mesh.Boundaries('x'))\
                                                            - delta*(grad(u)*n)*v*ds(skeleton=True,definedon=~mesh.Boundaries('x')) \
                                                            + (1-delta)*1j*k*u*v*ds(skeleton=True,definedon=~mesh.Boundaries('x'))


blfA+=(blfA_Ak)
blfA+=(blfA_Bk)


blfA.Assemble();


for refine_cnt in range(0,num_refines):
    mesh.Refine()
    ng_space.Update();
    blfA.Assemble();

Ainv=blfA.mat.Inverse();
Ainvop=NgOperator(Ainv,isComplex=True);
print("done assembling FEM")



trace_space, trace_matrix,dn_trace_matrix =  L2_trace(ng_space,also_give_dn=True,weight1=(1-delta),weight2=delta);
p_space= bempp.api.function_space(trace_space.grid, "P", order )
dp_space= bempp.api.function_space(p_space.grid, "DP", order-1 )


print("FEM dofs: {0}".format(ng_space.ndof))
print("BEM dofs: {0}".format(dp_space.global_dof_count))

totaldofs=ng_space.ndof+ dp_space.global_dof_count + p_space.global_dof_count

#increase the quadrature order and accuracy a bit. Otherwise higher order does not work
bempp.api.enable_console_logging()


assembler="dense"

slp_dd= bempp.api.operators.boundary.helmholtz.single_layer(
    dp_space, p_space, dp_space, k,assembler=assembler);
slp_cd= bempp.api.operators.boundary.helmholtz.single_layer(p_space, p_space, dp_space, k,assembler=assembler)
slp_dc= bempp.api.operators.boundary.helmholtz.single_layer(dp_space, p_space, p_space, k,assembler=assembler)
slp_cc= bempp.api.operators.boundary.helmholtz.single_layer(p_space, p_space, p_space, k,assembler=assembler)



hyp=bempp.api.operators.boundary.helmholtz.hypersingular(
    p_space,p_space,p_space,k,assembler=assembler); 

M_cc=bempp.api.operators.boundary.sparse.identity( p_space, p_space,p_space)
M_cd=bempp.api.operators.boundary.sparse.identity( p_space, p_space,dp_space)
M_traced=bempp.api.operators.boundary.sparse.identity( trace_space, p_space,dp_space)
M_dc=bempp.api.operators.boundary.sparse.identity( dp_space,p_space,p_space)
M_dtrace=bempp.api.operators.boundary.sparse.identity( dp_space,trace_space,trace_space)
M_dd=bempp.api.operators.boundary.sparse.identity( dp_space, p_space,dp_space)



dlp_cd = bempp.api.operators.boundary.helmholtz.double_layer(
    p_space, p_space, dp_space, k,assembler=assembler)

dlp_cc = bempp.api.operators.boundary.helmholtz.double_layer(
    p_space, p_space, p_space, k,assembler=assembler)


adlp_dc = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
    dp_space, p_space, p_space, k,assembler=assembler)

adlp_cc = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
    p_space, p_space, p_space, k,assembler=assembler)


Bk=-hyp + 1j*eta*(0.5*M_cc - dlp_cc)
Akp1=0.5*M_cc+ adlp_cc- 1j*eta*slp_cc
Akp2=0.5*M_dc+ adlp_dc- 1j*eta*slp_dc



Ak=0.5*M_cc+dlp_cc - 1j*eta*slp_cc;


u_ex=ngs.sin(k*x)*ngs.cos(k*y);
du_ex=[k*cos(k*x)*cos(k*y),-k*sin(k*x)*sin(k*y),ngs.CoefficientFunction(0)];
laplace_uex=-2*( (k)**2)*u_ex


f=ngs.LinearForm(ng_space);
lfi=ngs.SymbolicLFI((-laplace_uex- k**2*u_ex)*v);
f+=lfi

f.Assemble();


@bempp.api.complex_callable
def dirichlet_fun(x, n, domain_index, result):    
    r=np.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
    c=np.sin(k*x[0])*np.cos(k*x[1])


    result[0] = c-np.exp(1j*k*r)/r ;

@bempp.api.complex_callable
def impedance_fun(x, n, domain_index, result):
    r=np.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])

    dne_fac=1j*np.exp(1j*k*r)*(k*r+1j)/(r*r*r);
    dne=dne_fac*(x[0]*n[0]+x[1]*n[1]+x[2]*n[2]);

    du_ex=[k*np.cos(k*x[0])*np.cos(k*x[1]),-k*np.sin(k*x[0])*np.sin(k*x[1]),0];

    dni=du_ex[0]*n[0]+du_ex[1]*n[1]+du_ex[2]*n[2];


    c=np.sin(k*x[0])*np.cos(k*x[1])
    df= c-np.exp(1j*k*r)/r ;

    result[0] = dni-dne + 1j*k*df;




from scipy.sparse.linalg.interface import LinearOperator
blocks = np.ndarray([2,2],dtype='object');

mixed_dg_trace=-ikinv*dn_trace_matrix+trace_matrix;
trace_op = LinearOperator(trace_matrix.shape, lambda x:mixed_dg_trace*x)


##compute the new right hand side after eliminating u
#y2=f.vec.CreateVector();
y2=Ainvop*f.vec.FV().NumPy(); #TODO CHECK this one
ht=mixed_dg_trace.dot(y2);
ht=M_traced.weak_form()*ht;


B=mixed_dg_trace.T * M_dtrace.weak_form().A;
Bop=LinearOperator(B.shape, lambda x:B*x)
E=M_traced.weak_form().A*mixed_dg_trace
Eop=LinearOperator(E.shape, lambda x:E*x)


bem_block=np.ndarray([2,2],dtype='object');

bem_block[0,1] = - Akp2
bem_block[0,0]= Bk + 1j*k*Akp1
bem_block[1,0]=-(0.5*M_cd+dlp_cd + 1j*k*slp_cd)


##delta=d0*k*h/order
weight_matrix=d0*k*order*weight_bem_function(dp_space,1);

weight_operator=LinearOperator(M_dd.weak_form().shape, lambda x:weight_matrix*x)


print("setting up bem")
blocks[0,1] = bem_block[0,1].weak_form();
blocks[0,0] = bem_block[0,0].weak_form();
blocks[1,0] = bem_block[1,0].weak_form()
blocks[1,1] = slp_dd.weak_form() + ikinv*weight_operator@M_dd.weak_form() + Eop*Ainvop*Bop



print("done setting up system ");

# The rhs from the BEM
d_fun=bempp.api.GridFunction(p_space,dual_space=dp_space, fun=dirichlet_fun);
imp_fun=bempp.api.GridFunction(dp_space,fun=impedance_fun);
rhs_bem1 = -Akp2.weak_form()*imp_fun.coefficients;
rhs_bem2=  d_fun.projections(dp_space) + (slp_dd.weak_form()*imp_fun.coefficients) - ht;
# The combined rhs
rhs = np.concatenate([rhs_bem1,rhs_bem2])





print("using an iterative solver with block diagonal Prec")
from bempp.api.assembly.discrete_boundary_operator import InverseSparseDiscreteBoundaryOperator
prec_blocks = np.ndarray([2,2],dtype='object');
prec_blocks[0,0]=InverseSparseDiscreteBoundaryOperator(M_cc.weak_form())    
prec_blocks[1,1]=InverseSparseDiscreteBoundaryOperator(M_dd.weak_form())    

from  numpy import bmat
from scipy.sparse.linalg import spsolve;
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import lu_factor, lu_solve


from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
blocked = BlockedDiscreteOperator(blocks)
P = BlockedDiscreteOperator(prec_blocks)
# Create a callback function to count the number of iterations
it_count = 0
def count_iterations(x):
    global it_count
    it_count += 1

print("solving...")
from scipy.sparse.linalg import gmres,bicgstab
#with TaskManager():
soln, info = gmres(blocked, rhs, M=P, tol=1E-06, callback=count_iterations,restart=5000)
print("info: ",info)
it_counts.append(it_count)

print("solved");
print("Number of iterations: {0}".format(it_count))


print("splitting solution1")
#soln_fem = soln[:ng_space.ndof]

print("splitting solution2")
soln_bem = soln

print("splitting solution3")
soln_phi=soln_bem[:p_space.global_dof_count]
print("splitting solution4")
soln_m=soln_bem[p_space.global_dof_count:]
print("splitting solution done")



##for computing the errors, we crank up the quadrature orders
# Store the FEM solution
gfu = ngs.GridFunction(ng_space)
tmp=gfu.vec.CreateVector();
tmp.data=f.vec;

print("shape:",B.shape,soln_m.shape, tmp.FV().NumPy().shape)
tmp.FV().NumPy()[:] += Bop*soln_m;
gfu.vec.FV().NumPy()[:] = Ainvop*tmp.FV().NumPy();#np.ascontiguousarray(np.real(soln_fem))

print("l2err")
err_l2= sqrt(abs(ngs.Integrate((gfu-u_ex)*(gfu-u_ex),mesh,order=4*order+4)) /
             abs(ngs.Integrate((u_ex)*(u_ex),mesh,order=4*order+4))
);

print("h1err")
cfd=ngs.CoefficientFunction((du_ex[0],du_ex[1],du_ex[2]));

err_h1= sqrt(
    abs(ngs.Integrate((gfu.Deriv()-cfd)**2,mesh,order=4*order+4)) /
    abs(ngs.Integrate((cfd)**2,mesh,order=4*order+4))
);

print("h12err")
def phi_ex(pnt):
    r=sqrt(pnt[0]*pnt[0]+pnt[1]*pnt[1]+pnt[2]*pnt[2]);
    return exp(1j*k*r)/r;

phi=bempp.api.GridFunction(p_space,coefficients=soln_phi)
err_phi=0#phi.relative_error(phi_ex);


print("h-12err")

#normal=ngs.specialcf.normal(3);
def m_ex(x,n):
    #p=mesh(x[0],x[1],x[2],ngs.BND);
    #n=normal(p);
    u_ex=sin(k*x[0])*cos(k*x[1]);
    du_ex=[k*cos(k*x[0])*cos(k*x[1]),-k*sin(k*x[0])*sin(k*x[1]),0];

    dni=du_ex[0]*n[0]+du_ex[1]*n[1]+du_ex[2]*n[2];

    return (dni + 1j*k*u_ex);


m=bempp.api.GridFunction(dp_space,coefficients=soln_m)
err_m=0#m.relative_error2(m_ex);



print("L^2-error",err_l2, "h1-error:", err_h1, "error-phi: ",err_phi,"error-m",err_m," dofs: ",totaldofs, "k:",k, " n_iter: ",it_count)
ngs.Draw(u_ex,mesh,'uex')
ngs.Draw(gfu,mesh,'u')

