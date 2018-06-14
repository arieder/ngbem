def HCurl_trace(ng_space, boundaries=None):
    """
    Return the lowest oder Nedelec trace operator.

    This function returns a pair (space, trace_matrix),
    where space is a BEM++ space object and trace_matrix is the corresponding
    matrix that maps the coefficients of a NGsolve function to its boundary
    trace coefficients in the corresponding BEM++ space.
    """


    import ngsolve as ngs;
    from ngbem import surface_mesh_from_ng


    if(ng_space.type != 'hcurlho' or ng_space.globalorder!=0):
        raise ValueError("ng_space must be a valid  lowest order HCurl space")

    import ngsolve as ngs;
    from bempp.api import function_space, grid_from_element_data, ALL
    import bempp.api
    import numpy as np

    mesh = ng_space.mesh;

    [bm_coords,bm_cells,domain_indices,bm_nodes] = surface_mesh_from_ng(mesh);

    bempp_boundary_grid = grid_from_element_data(
        bm_coords, bm_cells,domain_indices)

    # First get trace space
    space = function_space(bempp_boundary_grid, "RT", 0)

    # Now compute the mapping from NGSolve dofs to BEM++ dofs.
    bem_elements = list(bempp_boundary_grid.leaf_view.entity_iterator(0))

    n_bem=space.global_dof_count;
    n_ng=ng_space.ndof;

    k=1
    print("doing order ",k,"dofs=",space.global_dof_count);
    nd=3 ##number of degrees of freedom on the reference element
    from math import sqrt;

    #setup the lagrange interpolation points for the BEM++ local basis

    eval_pts=np.asarray([[0.5,0],[0,0.5],[0.5,0.5]]).T;
    local_normals=np.asarray([[0,-1],[-1,0],[1,1]]);

    local_tangentials=np.asarray([[-1,1],[1,0],[0,-1]]); #the tangential vectors used by ngsolve
    ##local_tangentials=np.asarray([[-1,1],[1,0],[0,-1]]); #the tangential vectors used by ngsolve


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

        bem_global_dofs, bem_weights = space.get_global_dofs(bem_el, dof_weights=True)


        ngshape=ng_space.GetFE(ngs.ElementId(el));

        #evaluate the NGSolve basis in the midpoint of edges times the tangential vectors
        # this corresponds to the functionals assigned to the Nedelec basis functions,
        # and most notably gives how the local DOFs are assigned to the edges (including the weights)
        # local_bem_to_ng should be  a permutation matrix up to signs
        for j in range(0,nd):
            tx=TA.dot(eval_pts[:,j])+Tb; # transfer to NGSolves reference element
            tangential=local_tangentials[j]; #the tangentials are already stored transformed


            uj=ngshape.CalcShape(tx[0],tx[1],0);

            for l in range(0,nd):
                local_bem_to_ng[l,j]=np.dot(uj[l,:],tangential);


        local_ndofs=len(ng_dofs)
        assert(nd==local_ndofs)

        bem_global_dofs, bem_weights = space.get_global_dofs(bem_el, dof_weights=True)

        assert(len(bem_global_dofs) == len(ng_dofs));

        for i in range(0,local_ndofs):
            gbid=bem_global_dofs[i];
            if(todo[gbid]==1): ##we havent dealt with this index before
                for j in range(0,local_ndofs):
                    gngid=ng_dofs[j];
                    iis[idCnt]=gbid;
                    ijs[idCnt]=gngid;
                    data[idCnt]=bem_weights[i]*local_bem_to_ng[j,i];

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
