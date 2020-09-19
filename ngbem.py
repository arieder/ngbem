"""Implement the mapping from NGSolve P1 to BEM++ P1 functions."""

def surface_mesh_from_ng(mesh,bembnd=None):
    import numpy as np;
    import ngsolve as ngs;

    if bembnd is None:
        bembnd = mesh.Boundaries(".*")
    mask = bembnd.Mask()

    nv=mesh.nv
    nse=mesh.GetNE(ngs.BND)

    surfaceNodeToNode=np.full(nv,-1,np.int)
    nodeToSurfaceNode=np.full(nv,-1,np.int)

    surfaceNodeIdx=0

    #extract surface vertices
    pnts = []
    els = []
    dominds = []
    for se in mesh.Elements(ngs.BND):
        if mask[se.index]:
            for vert in se.vertices:
                if(nodeToSurfaceNode[vert.nr] == -1): #found one we hadnt before
                    nodeToSurfaceNode[vert.nr]=surfaceNodeIdx;
                    surfaceNodeToNode[surfaceNodeIdx]=vert.nr;
                
                    surfaceNodeIdx+=1;
                    pnts.append (mesh[vert].point)

            els.append ( [nodeToSurfaceNode[v.nr] for v in se.vertices] )
            dominds.append(se.index)


    #forget about the non-surface vertices
    nv=surfaceNodeIdx;
    print("got ",nv," surface vertices")
    # vertices=np.ndarray([3,nv],dtype=np.float64);

    # ngmesh = mesh.ngmesh
    # from netgen.meshing import PointId
    # for i in range(0,nv):
    #     ngp=ngmesh.Points()[PointId(surfaceNodeToNode[i]+1)];
    #     vertices[:,i]=ngp.p

    vertices = np.array(list( zip(*pnts) ))

    
    # domain_indices=np.zeros([nse],dtype=np.int);
    # elements=np.zeros([3,nse],dtype=np.int);
    # i=0;
    # for el in mesh.Elements(ngs.BND):
    #   if mask[se.index]:        
    #     j=0;
    #     domain_indices[i]=el.index;
    #     for p in el.vertices:
    #         elements[j,i]=nodeToSurfaceNode[p.nr]
    #         j+=1;
    #     i+=1;

    # filter only valid els, should be done from the beginning
    # els2 = []
    # for i in range(nse):
    # if elements[0,i] >= 0:
    # els2.append( (elements[0,i], elements[1,i], elements[2,i]) )
    elements = np.array(list(zip(*els)))
    domain_indices = np.array(dominds, dtype=np.int)
    
    return [vertices,elements,domain_indices, surfaceNodeToNode[0:nv]]


def bempp_grid_from_ng(mesh,bembnd=None):
    #from bempp.api import grid_from_element_data
    from bempp.api import Grid as grid_from_element_data
    [bm_coords,bm_cells,domain_indices,bm_nodes] = surface_mesh_from_ng(mesh, bembnd)

    bempp_boundary_grid = grid_from_element_data(
        bm_coords, bm_cells,domain_indices)
    
    return bempp_boundary_grid



def ng_surface_trace(ng_space,bempp_boundary_grid=None, bembnd=None):
    """
    Returns the trace operator for a NGSolve FESpace exporting surface elemenets.
    This can'be H^1 or SurfaceL2

    This function returns a pair (space, trace_matrix),
    where space is a BEM++ space object and trace_matrix is the corresponding
    matrix that maps the coefficients of a NGsolve function to its boundary
    trace coefficients in the corresponding BEM++ space.
    """

    import ngsolve as ngs;


    # if(ng_space.type != 'h1ho' and ng_space.type!='l2surf'):
    if ng_space.type not in ['h1ho', 'l2surf', 'wrapped-l2surf']:
        raise ValueError("ng_space must be a valid H1 or L2 surface space")

    import ngsolve as ngs;
    from bempp.api import function_space, ALL
    from bempp.api import Grid as grid_from_element_data
    import numpy as np

    mesh = ng_space.mesh;
    if bembnd is None:
        bembnd = mesh.Boundaries(".*")
    mask = bembnd.Mask()

    if(bempp_boundary_grid==None):
        [bm_coords,bm_cells,domain_indices,bm_nodes] = surface_mesh_from_ng(mesh, bembnd);
        bempp_boundary_grid = grid_from_element_data(
            bm_coords, bm_cells,domain_indices)

    # First get trace space
    if(ng_space.type=='h1ho'):
        space = function_space(bempp_boundary_grid, "P", ng_space.globalorder)
    else:
        space = function_space(bempp_boundary_grid, "DP", ng_space.globalorder)

    # Now compute the mapping from NGSolve dofs to BEM++ dofs.
    bem_elements = bempp_boundary_grid.elements#list(bempp_boundary_grid.leaf_view.entity_iterator(0))

    n_bem=space.global_dof_count;
    n_ng=ng_space.ndof;

    k=ng_space.globalorder
    print("doing order ",k);
    nd=int((k+1)*(k+2)/2) ##number of degrees of freedom on the reference element


    #setup the lagrange interpolation points for the BEM++ local basis
    eval_pts=np.zeros([2,nd]);
    pt_id=0;
    for j in range(0,k+1):
        if(k==0):
            yi=0.5;
        else:
            yi=j/k;
        for i in range(0,k-j+1):
            if(k==0):
                xi=0.5
            else:
                xi=i/k;
            eval_pts[:,pt_id]=[xi,yi];
            pt_id+=1;

    #leaf=space.grid.leaf_view;
    el0=bem_elements[:,0]#leaf.element_from_index(0);

    bem_shape=space.shapeset#(el0);
    vj=bem_shape.evaluate(eval_pts)
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
      if mask[el.index]:
        bem_el=bem_elements[:,elId];
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

        #bem_global_dofs, bem_weights = space.get_global_dofs(bem_el, dof_weights=True)
        bem_global_dofs =  space.local2global[elId]
        bem_weights = space.local_multipliers[elId]

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

def L2_trace(ng_space,bempp_boundary_grid=None,also_give_dn=False,weight1=1,weight2=1):
    from ngsolve import FacetFESpace, SurfaceL2,comp,BND, InnerProduct, specialcf
    if(ng_space.type != 'l2ho'):
        raise ValueError("ng_space must be a valid L2 space")

    mesh=ng_space.mesh
    F = FacetFESpace(mesh, order=ng_space.globalorder, dirichlet=".*",complex=ng_space.is_complex)
    S = SurfaceL2(mesh, order=ng_space.globalorder,complex=ng_space.is_complex)
    E1 = comp.ConvertOperator(spacea=ng_space, spaceb=F,trial_cf=(weight1)*ng_space.TrialFunction())
    E2 = comp.ConvertOperator(spacea=F, spaceb=S, vb=BND)

    
    E = to_sparse_matrix(E2) @ to_sparse_matrix(E1)


    #from ngsolve import  GridFunction, specialcf
    #from ngsolve.fem import  LoggingCF
    #gfu=GridFunction(F)
    #gfu.Set (  LoggingCF ( specialcf.mesh_size,logfile="bla.log" ),BND )


    [bem_space, trace_matrix]=ng_surface_trace(S,bempp_boundary_grid)

    if(also_give_dn):
        V2S_1 = comp.ConvertOperator(spacea=ng_space, spaceb=F, trial_cf=weight2*InnerProduct(ng_space.TrialFunction().Deriv(), specialcf.normal(mesh.dim)))
        V2S_2 = comp.ConvertOperator(spacea=F, spaceb=S, vb=BND)
        EN= to_sparse_matrix(V2S_2) @ to_sparse_matrix(V2S_1)

        return [bem_space, trace_matrix @ E, trace_matrix @ EN]
    else:
        return [bem_space, trace_matrix @ E]
        
    

def to_sparse_matrix(A):
    rows,cols,vals = A.COO()
    import scipy.sparse as sp

    B = sp.csr_matrix((vals,(rows,cols)),shape=(A.height,A.width))
    return B


def weight_bem_function(space,power):
    import numpy as np

    n_bem=space.global_dof_count
    todo=np.ones([n_bem]); #stores  which indices we already visited                                                                                                                                                                        
                                   

    iis=np.zeros([n_bem],dtype=np.int64);
    ijs=np.zeros([n_bem],dtype=np.int64);
    data=np.zeros([n_bem]);

    idCnt=0;

    grid=space.grid
    bem_elements = grid.elements
    
    for elId  in range(0,bem_elements.shape[1]):
        bem_global_dofs =  space.local2global[elId]
        bem_weights = space.local_multipliers[elId]

        
        corners=grid.vertices[:,bem_elements[:,elId]]
        #midpoint=(1.0/3.0)*(corners[:,0]+corners[:,1]+corners[:,2])                                                                                                                                                                                                   bem_el.geometry.volume        
        vol=np.linalg.norm(np.cross(corners[:,1]-corners[:,0],corners[:,2]-corners[:,0]))


        #print("v:",elId,np.power(vol,1/2))
        local_weight=pow(vol,0.5*power)
        #print("hi",local_weight)

        for i in range(0,len(bem_global_dofs)):
            gbid=bem_global_dofs[i];

            if(todo[gbid]==1): ##we havent dealt with this index before                                                                                                                                                                                                        
                iis[idCnt]=gbid;
                ijs[idCnt]=gbid;
                #print("looK:",bem_weights,local_weight)
                data[idCnt]=local_weight*bem_weights[i]

                idCnt+=1;
                todo[gbid]-=1;

    # build up the sparse matrix containing our transformation                                                                                                                                                                                                                 
    from scipy.sparse import coo_matrix

    matrix=coo_matrix((data,(iis,ijs)),shape=(space.global_dof_count,space.global_dof_count));
    return matrix
    


from scipy.sparse.linalg.interface import LinearOperator as _LinearOperator


class NgOperator(_LinearOperator):
    """provides a LinearOperator interface for NGSolves BilinearForm"""
    def __init__(self, blfOrMat, blf2=None, isComplex=False):
        from ngsolve import BaseVector,BilinearForm;
        """blf...the bilinear form to be applied,
        blf2 (optional)... a bilinear form of the same shape as blf,  useful if blf doesn't implement height and width"""
        if(isinstance(blfOrMat,BilinearForm)):
            self.blf=blfOrMat;
            self.mat=self.blf.mat
            if(blf2==None):
                self.tmp1 = BaseVector(self.blf.mat.width, isComplex);
                self.tmp2 = BaseVector(self.blf.mat.width, isComplex);
            else:
                self.tmp1 = BaseVector(blf2.mat.width, isComplex);
                self.tmp2 = BaseVector(blf2.mat.width, isComplex);
            self.shape=(self.blf.mat.height,self.blf.mat.width)
        else:
            self.mat=blfOrMat;
            if(blf2==None):
                self.tmp1 = BaseVector(self.mat.width, isComplex);
                self.tmp2 = BaseVector(self.mat.width, isComplex);
            else:
                self.tmp1 = BaseVector(blf2.mat.width, isComplex);
                self.tmp2 = BaseVector(blf2.mat.width, isComplex);
            self.shape=(self.mat.height,self.mat.width)

        self.dtype=self.tmp1.FV().NumPy().dtype;
        print("dtype=",self.dtype)


    def _matvec(self,v):
        import numpy as np
        self.tmp1.FV().NumPy()[:] = v.reshape(v.shape[0]);
        self.mat.Mult(self.tmp1,self.tmp2);

        return self.tmp2.FV().NumPy()
    def _matmat(self,vec):
        print('matmat not implemented')

    def tocsc(self):
        import scipy.sparse as sp
        rows,cols,vals = self.mat.COO()
        Acsc = sp.csc_matrix((vals,(rows,cols)))
        return Acsc;


#for compatibility reasons
def H1_trace(ng_space,bempp_surface_grid=None, bembnd=None):
    return ng_surface_trace(ng_space,bempp_surface_grid, bembnd);
    
def ng_to_bempp_trace(ng_space,bempp_surface_grid=None, bembnd=None):
    if(ng_space.type=='h1ho'):
        return H1_trace(ng_space,bempp_surface_grid, bembnd)
    if(ng_space.type=='l2ho'):
        return L2_trace(ng_space,bempp_surface_grid, bembnd)
    elif(ng_space.type=='l2surf' or ng_space.type=='wrapped-l2surf'):
        return ng_surface_trace(ng_space,bempp_surface_grid, bembnd);
    elif(ng_space.type=='hcurlho'):
        from maxwell_ngbem import HCurl_trace;
        return HCurl_trace(ng_space)
