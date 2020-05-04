from __future__ import division,print_function

import bempp.api
import numpy as np
from math import pi
from IPython import embed
from scipy.sparse.linalg import LinearOperator
from itertools import product

from matplotlib import pylab as plt

import ngbem
import maxwell_ngbem

from time import time
the_time = str(time())

k = 1.

ns = []
errs = []
err2s = []


import netgen.csg;
import ngsolve as ngs;
cube = netgen.csg.OrthoBrick( netgen.csg.Pnt(0,0,0), netgen.csg.Pnt(1,1,1) )

geo = netgen.csg.CSGeometry()
geo.Add (cube)

bempp.api.global_parameters.hmat.eps=1E-05;
bempp.api.global_parameters.hmat.max_rank=4096;



for n in range(1,12):
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=1/n))

    epsilon = 1.
    mu = 1.
    mu_inv = 1./mu
    omega = k/np.sqrt(mu*epsilon)

    def incident_field(x):
        return np.array([np.exp(1j * k * x[2]), 0.*x[2], 0.*x[2]])

    def curl_incident_field(x):
        return np.array([0.*x[2], np.exp(1j * k * x[2]), 0.*x[2]])

    def tangential_trace(x, n, domain_index, result):
        result[:] = np.cross(incident_field(x), n, axis=0)

    def neumann_trace(x, n, domain_index, result):
        result[:] = np.cross(curl_incident_field(x), n, axis=0)

    def zero(x, n, domain_index, result):
        result[:] = np.zeros(3)

    # Define function spaces
    fem_space=ngs.HCurl(mesh,order=0,complex=True);

    trace_space,trace_matrix  = maxwell_ngbem.HCurl_trace(fem_space);
    trace_op = LinearOperator(trace_matrix.shape, lambda x:trace_matrix*x)
    trace_op_adj = LinearOperator(trace_matrix.T.shape, lambda x:trace_matrix.T*x)
    grid = trace_space.grid

    rt_space = bempp.api.function_space(grid,"RT",0) ##why barycentric?
    nc_space = bempp.api.function_space(grid, "NC", 0) #why barycentric

    bc_space = bempp.api.function_space(grid,"RT",0)
    #bnc_space = bempp.api.function_space(grid,"B-NC",0)
    rbc_space = bempp.api.function_space(grid,"NC",0)

    fem_size = fem_space.ndof
    trace_size = rt_space.global_dof_count
    bem_size = rt_space.grid.leaf_view.entity_count(2)

    # Define operators
    E = bempp.api.operators.boundary.maxwell.electric_field(bc_space,rt_space,rbc_space,k)
    Id2 = bempp.api.operators.boundary.sparse.identity(rt_space,rt_space,rbc_space)
    H = bempp.api.operators.boundary.maxwell.magnetic_field(rt_space,rt_space,rbc_space,k)

    Id = bempp.api.operators.boundary.sparse.identity(bc_space,bc_space,nc_space)

    Id.weak_form()
    #FEniCS
    tu = fem_space.TrialFunction();
    tv = fem_space.TestFunction();

    # Make right hand side
    e_inc = bempp.api.GridFunction(rt_space,dual_space=rbc_space,fun=tangential_trace)
    e_N_inc = bempp.api.GridFunction(bc_space,dual_space=nc_space,fun=neumann_trace)
    print(1)
    f_upper = 1j*k/mu * trace_matrix.T * (Id.weak_form() * e_N_inc.coefficients)
    print(2)
    f_lower = (.5*Id2+H).weak_form() * e_inc.coefficients
    print(3)
    f_0 = np.concatenate([f_upper,f_lower])

    # Build BlockedLinearOperator
    blocks = [[None,None],[None,None]]

    A  = ngs.BilinearForm(fem_space);
    A+= ngs.SymbolicBFI(mu_inv*ngs.curl(tu)*ngs.curl(tv));
    A+= ngs.SymbolicBFI(-omega**2*epsilon*tu*tv);


    A.Assemble();


    blocks[0][0] = ngbem.NgOperator(A,isComplex=True);

    print("done assembling fem")
    blocks[0][1] = -1j*k/mu * trace_op_adj * Id.weak_form()
    blocks[1][0] = (.5*Id2 + H).weak_form() * trace_op
    blocks[1][1] = E.weak_form()


    precond_blocks = [[None,None],[None,None]]

    precond_blocks[0][0]=np.identity(fem_size);#ngbem.NgOperator(fem_prec,A);
    precond_blocks[1][1]=np.identity(E.weak_form().shape[0])#bem_prec;

    print("done assembling bem")
    blocked = bempp.api.BlockedDiscreteOperator(blocks)
    precond= bempp.api.BlockedDiscreteOperator(precond_blocks)

    from scipy.sparse.linalg import gmres as solver
    it_count = 0
    def iteration_counter(x):
        global it_count
        it_count += 1

    print("solving")
    soln,err = solver(blocked,f_0,M=precond,callback=iteration_counter,tol=1E-8,maxiter=50000,restart=2000)

    print("converged after ",it_count,"steps")

    soln_fem = soln[:fem_size]
    soln_bem = soln[fem_size:]

    u=ngs.GridFunction(fem_space)

    u.vec.FV().NumPy()[:]=np.array(soln_fem.real,dtype=np.float_)

    #u_im=dolfin.Function(fenics_space)
    #u_im.vector()[:]=np.array(soln_fem.imag,dtype=np.float_)

    bem_soln = bempp.api.GridFunction(bc_space,coefficients=soln_bem)

    ns.append(fem_size)

    actual0 = bempp.api.GridFunction(bc_space, fun=zero)

    errs.append((actual0-bem_soln).l2_norm())

    actual_u = ngs.CoefficientFunction((ngs.cos(k*ngs.z),0,0));
    actual_curl_u = ngs.CoefficientFunction((0,-k*ngs.sin(k*ngs.z),0));

    actual_sol = np.concatenate([soln_fem, soln_bem])

    from math import sqrt;
    u_H1 = sqrt(abs(ngs.Integrate((actual_u-u)*(actual_u-u)+ (actual_curl_u -u.Deriv())**2,mesh,order=10)));
    err2s.append(u_H1)

    print("ns: ",ns)
    print("Hcurl error:",err2s)
    print("BEM error",errs)


    plt.plot(ns,err2s,"ro-")
    plt.plot(ns,errs,"bo-")
    plt.legend(["FEM part","BEM part"])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("number of FEM DOFs")
    plt.ylabel("Error")
    plt.savefig("output/conv.png")
    plt.clf()

