
# We begin by importing Dolfin, the ngsolve python library, Bempp and NumPy.

# In[1]:

from mpi4py import MPI
import ngsolve as ngs
import bempp.api
import numpy as np


# Next, we set the wavenumber ``k`` and the direction ``d`` of the incoming wave.

# In[2]:


k = 6.
d = np.array([1., 1., 1])
d /= np.linalg.norm(d)



from netgen.csg import *
geo1 = CSGeometry()
geo1.Add (OrthoBrick(Pnt(0,0,0),Pnt(1,1,1)))
m1 = geo1.GenerateMesh (maxh=0.1)

mesh = ngs.Mesh(m1)

# Next, we make the NGSolve and Bempp function spaces.
# 
# The function ``H1_trace`` will extract the trace space from the ngsovle space and create the matrix ``trace_matrix``, which maps between the dofs (degrees of freedom) in NGSolve and Bempp.

# In[4]:

from  ngbem import *
order=2;
ng_space=ngs.H1(mesh,order=order,complex=True)


trace_space, trace_matrix =  H1_trace(ng_space);
bempp_space = bempp.api.function_space(trace_space.grid, "DP", order-1)

print("FEM dofs: {0}".format(mesh.nv))
print("BEM dofs: {0}".format(bempp_space.global_dof_count))





#increase the quadrature order and accuracy a bit. Otherwise higher order does not work
bempp.api.global_parameters.hmat.eps=1E-05;
bempp.api.global_parameters.hmat.max_rank=2048;

bempp.api.global_parameters.quadrature.double_singular += order + 3
bempp.api.global_parameters.quadrature.near.double_order += order + 3
bempp.api.global_parameters.quadrature.medium.double_order += order + 3
bempp.api.global_parameters.quadrature.far.double_order += order + 3
bempp.api.global_parameters.quadrature.near.single_order += order + 3
bempp.api.global_parameters.quadrature.medium.single_order += order + 3
bempp.api.global_parameters.quadrature.far.single_order += order + 3



# We create the boundary operators that we need.

# In[5]:


id_op = bempp.api.operators.boundary.sparse.identity(
    trace_space, bempp_space, bempp_space)
mass = bempp.api.operators.boundary.sparse.identity(
    bempp_space, bempp_space, trace_space)
dlp = bempp.api.operators.boundary.helmholtz.double_layer(
    trace_space, bempp_space, bempp_space, k)
slp = bempp.api.operators.boundary.helmholtz.single_layer(
    bempp_space, bempp_space, bempp_space, k)


# We create the NGSolve function spaces and the function (or in this case constant) ``n``.

# In[6]:


u,v=ng_space.TnT();

n = 0.5


# We make the vectors on the right hand side of the formulation.

# In[7]:


def u_inc(x, n, domain_index, result):
    result[0] = np.exp(1j * k * np.dot(x, d))
u_inc = bempp.api.GridFunction(bempp_space, fun=u_inc)

# The rhs from the FEM
rhs_fem = np.zeros(ng_space.ndof)
# The rhs from the BEM
rhs_bem = u_inc.projections(bempp_space)
# The combined rhs
rhs = np.concatenate([rhs_fem, rhs_bem])


# We are now ready to create a ``BlockedLinearOperator`` containing all four parts of the discretisation of
# $$
# \begin{bmatrix}
#     \mathsf{A}-k^2 \mathsf{M} & -\mathsf{M}_\Gamma\\
#     \tfrac{1}{2}\mathsf{Id}-\mathsf{K} & \mathsf{V}
# \end{bmatrix}.
# $$

# In[8]:



from scipy.sparse.linalg.interface import LinearOperator
blocks = [[None,None],[None,None]]

trace_op = LinearOperator(trace_matrix.shape, lambda x:trace_matrix*x)


blfA=ngs.BilinearForm(ng_space)
blfA+=ngs.SymbolicBFI(ngs.grad(u)*ngs.grad(v) - k**2 * n**2 * u * v)

c = ngs.Preconditioner(blfA, type="direct");

blfA.Assemble();

blocks[0][0] = NgOperator(blfA)
blocks[0][1] = -trace_matrix.T * mass.weak_form().sparse_operator
blocks[1][0] = (.5 * id_op - dlp).weak_form() * trace_op
blocks[1][1] = slp.weak_form()

blocked = bempp.api.BlockedDiscreteOperator(np.array(blocks))


# Next, we solve the system, then split the solution into the parts assosiated with u and &lambda;. For an efficient solve, preconditioning is required.

# In[9]:


from scipy.sparse.linalg import LinearOperator

# Compute the sparse inverse of the Helmholtz operator
# Although it is not a boundary operator we can use
# the SparseInverseDiscreteBoundaryOperator function from
# BEM++ to turn its LU decomposition into a linear operator.


P1 = NgOperator(c,blfA)

# For the Laplace slp we use a simple mass matrix preconditioner. 
# This is sufficient for smaller low-frequency problems.
P2 = bempp.api.InverseSparseDiscreteBoundaryOperator(
    bempp.api.operators.boundary.sparse.identity(
        bempp_space, bempp_space, bempp_space).weak_form())

# Create a block diagonal preconditioner object using the Scipy LinearOperator class
def apply_prec(x):
    """Apply the block diagonal preconditioner"""
    m1 = P1.shape[0]
    m2 = P2.shape[0]
    n1 = P1.shape[1]
    n2 = P2.shape[1]
    
    res1 = P1.dot(x[:n1])
    res2 = P2.dot(x[n1:])
    return np.concatenate([res1, res2])

p_shape = (P1.shape[0] + P2.shape[0], P1.shape[1] + P2.shape[1])
P = LinearOperator(p_shape, apply_prec, dtype=np.dtype('complex128'))

# Create a callback function to count the number of iterations
it_count = 0
def count_iterations(x):
    global it_count
    it_count += 1

print("solving...")
from scipy.sparse.linalg import gmres
soln, info = gmres(blocked, rhs, M=P, callback=count_iterations)

print("solved");
soln_fem = soln[:ng_space.ndof]
soln_bem = soln[ng_space.ndof:]

print("Number of iterations: {0}".format(it_count))


# Next, we make Dolfin and Bempp functions from the solution.

# In[10]:


# Store the real part of the FEM solution
gfu = ngs.GridFunction(ng_space)
gfu.vec.FV().NumPy()[:] = np.ascontiguousarray(np.real(soln_fem))

# Solution function with dirichlet data on the boundary
dirichlet_data = trace_matrix * soln_fem
dirichlet_fun = bempp.api.GridFunction(trace_space, coefficients=dirichlet_data)

# Solution function with Neumann data on the boundary
neumann_fun = bempp.api.GridFunction(bempp_space, coefficients=soln_bem)


# We now evaluate the solution on the slice $z=0.5$ and plot it. For the exterior domain, we use the respresentation formula
# 
# $$
# u^\text{s} = \mathcal{K}u-\mathcal{V}\frac{\partial u}{\partial \nu}
# $$
# 
# to evaluate the solution.

# In[11]:

# The next command ensures that plots are shown within the IPython notebook
#get_ipython().run_line_magic('matplotlib', 'inline')


# Reduce the H-matrix accuracy since the evaluation of potentials for plotting
# needs not be very accurate.
bempp.api.global_parameters.hmat.eps = 1E-2

Nx=200
Ny=200
xmin, xmax, ymin, ymax=[-1,3,-1,3]
plot_grid = np.mgrid[xmin:xmax:Nx*1j,ymin:ymax:Ny*1j]
points = np.vstack((plot_grid[0].ravel(),
                    plot_grid[1].ravel(),
                    np.array([0.5]*plot_grid[0].size)))
plot_me = np.zeros(points.shape[1], dtype=np.complex128)

x,y,z = points
bem_x = np.logical_not((x>0) * (x<1) * (y>0) * (y<1) * (z>0) * (z<1))

slp_pot= bempp.api.operators.potential.helmholtz.single_layer(
    bempp_space, points[:, bem_x], k)
dlp_pot= bempp.api.operators.potential.helmholtz.double_layer(
    trace_space, points[:, bem_x], k)

plot_me[bem_x] += np.exp(1j * k * (points[0, bem_x] * d[0]                                  + points[1, bem_x] * d[1]                                  + points[2, bem_x] * d[2]))
plot_me[bem_x] += dlp_pot.evaluate(dirichlet_fun).flat
plot_me[bem_x] -= slp_pot.evaluate(neumann_fun).flat

fem_points = points[:, np.logical_not(bem_x)].transpose()
fem_val = np.zeros(len(fem_points))
for p,point in enumerate(fem_points):
    fem_val[p] = gfu(point[0],point[1],point[2]).real

plot_me[np.logical_not(bem_x)] += fem_val

plot_me = plot_me.reshape((Nx, Ny))

plot_me = plot_me.transpose()[::-1]

# Plot the image
from matplotlib import pyplot as plt
fig=plt.figure(figsize=(10, 8))
plt.imshow(np.real(plot_me), extent=[xmin, xmax, ymin, ymax])
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.title("FEM-BEM Coupling for Helmholtz")

ngs.Draw(gfu)
plt.show()

