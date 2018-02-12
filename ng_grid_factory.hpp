#ifndef _NG_GRID_FACTORY_
#define _NG_GRID_FACTORY_

#include <boost/shared_ptr.hpp>


#include <grid/grid.hpp>
#include <comp.hpp>

namespace NgBem{

class NgGridFactory
{
public:
  static boost::shared_ptr<  Bempp::Grid > createBemppGridFromNgMesh( std::shared_ptr<ngcomp::MeshAccess> mesh, int domainIndex = -1 );

static void createConnectivityArraysFromNgMesh( std::shared_ptr<ngcomp::MeshAccess> mesh,
                                                int domainIndex,
                                                Bempp::Matrix<double>& vertices,
                                                Bempp::Matrix<int>& elementCorners,
                                                std::vector<int>& domainIndices
                                                );

};

}


#endif
