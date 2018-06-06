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

    domain_indices=np.zeros([nse],dtype=np.int);
    elements=np.zeros([3,nse],dtype=np.int);
    i=0;
    for el in mesh.Elements(ngs.BND):
        j=0;
        domain_indices[i]=el.index;
        for p in el.vertices:
            elements[j,i]=nodeToSurfaceNode[p.nr]
            j+=1;
        i+=1;


    return [vertices,elements,domain_indices, surfaceNodeToNode[0:nv]]




def H1_trace(ng_space):
    """
    Return the H^1-trace operator.

    This function returns a pair (space, trace_matrix),
    where space is a BEM++ space object and trace_matrix is the corresponding
    matrix that maps the coefficients of a NGsolve function to its boundary
    trace coefficients in the corresponding BEM++ space.
    """

    import ngsolve as ngs;


    if(ng_space.type != 'h1ho'):
        raise ValueError("ng_space must be a valid H1 space")

    import ngsolve as ngs;
    from bempp.api import function_space, grid_from_element_data, ALL
    import numpy as np

    mesh = ng_space.mesh;

    [bm_coords,bm_cells,domain_indices,bm_nodes] = surface_mesh_from_ng(mesh);

    bempp_boundary_grid = grid_from_element_data(
        bm_coords, bm_cells,domain_indices)

    # First get trace space
    space = function_space(bempp_boundary_grid, "P", ng_space.globalorder)

    # Now compute the mapping from NGSolve dofs to BEM++ dofs.
    bem_elements = list(bempp_boundary_grid.leaf_view.entity_iterator(0))

    n_bem=space.global_dof_count;
    n_ng=ng_space.ndof;

    k=ng_space.globalorder
    print("doing order ",k);
    nd=int((k+1)*(k+2)/2) ##number of degrees of freedom on the reference element


    #setup the lagrange interpolation points for the BEM++ local basis
    eval_pts=np.zeros([2,nd]);
    pt_id=0;
    for j in range(0,k+1):
        yi=j/k;
        for i in range(0,k-j+1):
            xi=i/k;
            eval_pts[:,pt_id]=[xi,yi];
            pt_id+=1;

    leaf=space.grid.leaf_view;
    el0=leaf.element_from_index(0);

    bem_shape=space.shapeset(el0);
    vj=bem_shape.evaluate(eval_pts,ALL)
    print("error due to eval",np.linalg.norm(vj-np.eye(nd)));


    local_bem_to_ng=np.zeros([nd,nd]);

    #NGSolve and BEM++ use a different reference element.
    #The map TA x + b does this transformatuib
    TA=np.asarray([[-1, -1], [1, 0]]);
    Tb=np.asarray([1, 0]);


    todo=np.ones([n_bem]); #stores  which indices we already visited

    iis=np.zeros([nd*n_bem],dtype=np.int64);
    ijs=np.zeros([nd*n_bem],dtype=np.int64);
    data=np.zeros([nd*n_bem]);

    elId=0;

    idCnt=0;
    #on each element, we map from ngsolve to the reference element,
    #do the local transformation there and then transform back to the global BEM++ dofs
    for el in ng_space.Elements(ngs.BND):
        bem_el=bem_elements[elId];
        ng_dofs=el.dofs

        ngshape=ng_space.GetFE(ngs.ElementId(el));


        #evaluate the NGSolve basis in the Lagrange points to get coefficients of the local transformation
        for j in range(0,nd):
            tx=TA.dot(eval_pts[:,j])+Tb;
            uj=ngshape.CalcShape(tx[0],tx[1],0);
            local_bem_to_ng[:,j]=uj;


        local_trafo=(local_bem_to_ng);


        local_ndofs=len(ng_dofs)
        assert(nd==local_ndofs)

        bem_global_dofs, bem_weights = space.get_global_dofs(bem_el, dof_weights=True)

        ng_weights=np.ones(len(ng_dofs));

        assert(len(bem_global_dofs) == len(ng_dofs));

        for i in range(0,local_ndofs):
            gbid=bem_global_dofs[i];
            if(todo[gbid]==1): ##we havent dealt with this index before
                for j in range(0,local_ndofs):
                    gngid=ng_dofs[j];
                    iis[idCnt]=gbid;
                    ijs[idCnt]=gngid;
                    data[idCnt]=bem_weights[i]*local_trafo[j,i];

                    idCnt+=1;
                todo[gbid]-=1;


        elId+=1;



    assert(np.count_nonzero(todo)==0)
    assert(idCnt==nd*n_bem)

    # build up the sparse matrix containing our transformation
    from scipy.sparse import coo_matrix

    trace_matrix=coo_matrix((data,(iis,ijs)),shape=(space.global_dof_count,ng_space.ndof));

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
        self.shape=(blf.mat.height,blf.mat.width)
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
