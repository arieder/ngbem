#include "ng_grid_factory.hpp"


#include <Eigen/Dense>

#include <grid/grid_factory.hpp>

namespace NgBem{

    boost::shared_ptr<  Bempp::Grid > NgGridFactory::createBemppGridFromNgMesh( std::shared_ptr<ngcomp::MeshAccess> mesh, int domainIndex )
{
    Bempp::Matrix<double> vertices;
    Bempp::Matrix<int> corners;

    Bempp::GridParameters params;
    params.topology = Bempp::GridParameters::TRIANGULAR;

    std::vector<int>  domainIndices;
    createConnectivityArraysFromNgMesh( mesh, domainIndex, vertices, corners,domainIndices );
    return  Bempp::GridFactory::createGridFromConnectivityArrays( params, vertices, corners, domainIndices );
}

void NgGridFactory::createConnectivityArraysFromNgMesh( std::shared_ptr<ngcomp::MeshAccess> mesh,
                                                        int domainIndex,
                                                        Bempp::Matrix<double>& vertices,
                                                        Bempp::Matrix<int>& elementCorners,
                                                        std::vector<int>& domainIndices )
{
    int np = mesh->GetNP();  /// number of points in mesh
    int nse = mesh->GetNSE();  /// number of surface elements (BC)
    int i,  k, l;

    ngstd::Array<int> surfaceNodeToNode( np );
    ngstd::Array<int> nodeToSurfaceNode( np );

    nodeToSurfaceNode=-1;
    surfaceNodeToNode=-1;
    int surfaceNodeIdx=0;
    bool found=false;

    std::cout<<"looking for surface vertices.."<<std::endl;
    //Slow as hell, but I don't care
    for ( int i=0;i<np;i++ )
    {
        found=false;
        for ( int j=0; j<nse;j++ )
        {
            const netgen::Ng_Element & el = mesh->GetSElement(j);

            if ( el.GetType() != NG_TRIG )
                throw Exception( "Only triangular boundary meshes supported" );


            for ( int k=0;k<el.points.Size();k++ )
            {
                if ( el.points[k]==i ) //this node is a surface element. keep it.
                {
                    nodeToSurfaceNode[i]=surfaceNodeIdx;
                    surfaceNodeToNode[surfaceNodeIdx]=i;

                    surfaceNodeIdx++;
                    found=true;
                    break;
                }
            }

            if ( found==true )
                break;
        }
    }

    //forget about the non-surface nodes
    np=surfaceNodeIdx;

    vertices.resize( 3,np );
    ngbla::Vec<3> p;
    for (i = 1; i <= np; i++)
    {
        const int nodeIdx=surfaceNodeToNode[i-1];
        mesh->GetPoint(nodeIdx, p);

        for ( int j=0;j<3;j++ )
            vertices( j, i-1 )=p( j );
    }

    std::cout<<"building elements.."<<std::endl;
    elementCorners.resize( 3, nse );
    domainIndices.resize(nse);
    int realnse=0;
    for (k = 0; k < nse; k++)
    {
        const netgen::Ng_Element & el = mesh->GetSElement(k);
        int in,out;
        mesh->GetSElNeighbouringDomains(k,in,out);
        //std::cout<<"k "<<k<<" in "<<in<<"  out "<<out<<std::endl;
        bool flip=false;
        bool drop=true;
        if(domainIndex!=-1)
        {
            if(in==domainIndex)
            {
                domainIndices[k]=in;
                drop=false;
            }
            if(out==domainIndex)  //flip
            {
                //std::cout<<"flipping"<<std::endl;
                flip=true;
                domainIndices[k]=out;
                drop=false;
            }
        }else
        {
            drop=false;
            domainIndices[k]=out;
        }

        if ( el.GetType() != NG_TRIG || el.points.Size()!=3)
            throw Exception( "Only triangular boundary meshes supported" );

        if(!drop)
        {
            static const int flipIds[]={2,1,0};
            for (l = 0; l < el.points.Size(); l++)
            {
                if(flip)
                    elementCorners( flipIds[l], realnse )=nodeToSurfaceNode[el.points[l] ];
                else
                    elementCorners( l, realnse )=nodeToSurfaceNode[el.points[l] ];

            }
            realnse++;
        }

    }

    //elementCorners.resize(3,realnse);
    //domainIndices.resize(realnse);
    std::cout<<"done generating bem++ grid"<<std::endl;
    std::cout<<"got "<<np<<"surface vertices"<<std::endl;
    std::cout<<"and "<<nse<<"surface triangles"<<std::endl;


}

}
