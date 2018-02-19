"""Implement the mapping from NGSolve P1 to BEM++ P1 functions."""

def surface_mesh_from_ng(mesh,domainIndex=0):
    import numpy as np;
    import ngsolve as ngs;


    nv=mesh.nv
    nse=mesh.GetNE(ngs.BND)

    surfaceNodeToNode=np.zeros(nv,np.int)
    nodeToSurfaceNode=np.zeros(nv,np.int)

    surfaceNodeToNode.fill(-1)
    nodeToSurfaceNode.fill(-1)

    surfaceNodeIdx=0

    #extract surface vertices
    for se in mesh.Elements(ngs.BND):
        for vert in se.vertices:
            if(nodeToSurfaceNode[vert.nr] == -1): #found one we hadnt before
                nodeToSurfaceNode[vert.nr]=surfaceNodeIdx;
                surfaceNodeToNode[surfaceNodeIdx]=vert.nr;

                surfaceNodeIdx+=1;


    #forget about the non-surface vertices
    nv=surfaceNodeIdx;
    print("got ",nv," surface vertices")
    vertices=np.ndarray([3,nv],dtype=np.float64);

    ngmesh = mesh.ngmesh
    from netgen.meshing import PointId
    for i in range(0,nv):
        ngp=ngmesh.Points()[PointId(surfaceNodeToNode[i]+1)];
        vertices[:,i]=ngp.p


    elements=np.zeros([3,nse],dtype=np.int);
    i=0;
    for el in mesh.Elements(ngs.BND):
        j=0;
        for p in el.vertices:
            elements[j,i]=nodeToSurfaceNode[p.nr]
            j+=1;
        i+=1;


    return [vertices,elements, surfaceNodeToNode[0:nv]]



def p1_dof_to_vertex_matrix(space):
    """Map from dofs to grid insertion vertex indices."""
    import numpy as np

    from scipy.sparse import coo_matrix

    grid = space.grid
    vertex_count = space.global_dof_count

    vertex_to_dof_map = np.zeros(vertex_count, dtype=np.int)

    for element in grid.leaf_view.entity_iterator(0):
        global_dofs = space.get_global_dofs(element)
        for ind, vertex in enumerate(element.sub_entity_iterator(2)):
            vertex_to_dof_map[grid.vertex_insertion_index(vertex)] = \
                global_dofs[ind]

    vertex_indices = np.arange(vertex_count)
    data = np.ones(vertex_count)
    return coo_matrix(
        (data, (vertex_indices, vertex_to_dof_map)), dtype='float64').tocsc()

#pylint: disable=too-many-locals

def p1_trace(ng_space):
    """
    Return the P1 trace operator.

    This function returns a pair (space, trace_matrix),
    where space is a BEM++ space object and trace_matrix is the corresponding
    matrix that maps the coefficients of a NGsolve function to its boundary
    trace coefficients in the corresponding BEM++ space.
    """

    import ngsolve as ngs;


    if(ng_space.globalorder != 1):
        raise ValueError("ng_space must be a p1 H1 space")

    import ngsolve as ngs;
    from bempp.api import function_space, grid_from_element_data
    import numpy as np

    mesh = ng_space.mesh;

    [bm_coords,bm_cells,bm_nodes] = surface_mesh_from_ng(mesh);

    bempp_boundary_grid = grid_from_element_data(
        bm_coords, bm_cells)

    # First get trace space
    space = function_space(bempp_boundary_grid, "P", 1)

    # Now compute the mapping from NGSolve dofs to BEM++ dofs.

    # First the BEM++ dofs from the boundary vertices
    from scipy.sparse import coo_matrix
    bempp_dofs_from_b_vertices = p1_dof_to_vertex_matrix(space).transpose()


    nsv=len(bm_nodes)
    # Now NGSolve vertices to boundary dofs
    b_vertices_from_vertices = coo_matrix(
        (np.ones(nsv), (np.arange(nsv), bm_nodes)),
        shape=(len(bm_nodes), mesh.nv),
        dtype='float64').tocsc()


    # Finally NGSolve dofs to vertices.
    dofmap=np.zeros([ng_space.ndof],dtype='int64')
    for el in ng_space.Elements(ngs.BND):
        dofs=el.dofs
        vs=el.vertices
        dofmap[dofs]=[vs[j].nr for j in range(0,len(vs))];



    vertices_from_ngs_dofs = coo_matrix(
        (np.ones(mesh.nv),
         (dofmap, np.arange(
             mesh.nv))),
        shape=(mesh.nv, mesh.nv),
        dtype='float64').tocsc()

    # Get trace matrix by multiplication
    trace_matrix = bempp_dofs_from_b_vertices * \
        b_vertices_from_vertices * vertices_from_ngs_dofs

    # Now return everything
    return space, trace_matrix


from scipy.sparse.linalg.interface import LinearOperator as _LinearOperator

class NgOperator(_LinearOperator):
    """provides a LinearOperator interface for NGSolves BilinearForm"""
    def __init__(self, blf, blf2=None):
        """blf...the bilinear form to be applied,
        blf2 (optional)... a bilinear form of the same shape as blf,  useful if blf doesn't implement height and width"""


        self.blf=blf;
        if(blf2==None):
            self.tmp1 = blf.mat.CreateColVector();
            self.tmp2 = blf.mat.CreateColVector();
        else:
            self.tmp1 = blf2.mat.CreateColVector();
            self.tmp2 = blf2.mat.CreateColVector();
        self.shape=[blf.mat.height,blf.mat.width]
        self.dtype=self.tmp1.FV().NumPy().dtype;

    def _matvec(self,v):
        import numpy as np
        self.tmp1.FV().NumPy()[:] = v.reshape(v.shape[0]);
        self.tmp2.data = self.blf.mat * self.tmp1
        return self.tmp2.FV().NumPy()
    def _matmat(self,vec):
        print('matmat not implemented')

    def tocsc(self):
        import scipy.sparse as sp
        rows,cols,vals = self.blf.mat.COO()
        Acsc = sp.csc_matrix((vals,(rows,cols)))
        return Acsc;
