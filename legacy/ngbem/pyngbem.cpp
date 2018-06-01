
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
//#include <pybind11/eigen.h>
namespace py=pybind11;

#include "ng_trace_space.hpp"
#include "ng_grid_factory.hpp"
#include "blockmatrix.hpp"

#include <assembly/discrete_boundary_operator.hpp>

//#include "py_util.hpp"

#include<python_ngstd.hpp>
#include<comp.hpp>

namespace NgBem{

template<typename BFT>
boost::shared_ptr<Bempp::Space<BFT> > ng_trace_space( const boost::shared_ptr<const Bempp::Grid>& grid,
                                                      PyObject* py_fespace)
{
    std::cout<<"making new trace"<<std::endl;
    std::shared_ptr<ngcomp::FESpace> pyfe=py::object(py_fespace,true).cast<std::shared_ptr<ngcomp::FESpace>  >();
    auto  space=boost::make_shared<NgBem::NgTraceSpace<BFT> >( grid, pyfe );
    std::cout<<"made the space"<<std::endl;
    return  space;
}



template<typename BFT>
PyObject*  c_bempp_to_ng_map( const boost::shared_ptr<const Bempp::Space<BFT> > & space )
{
    auto ng=boost::dynamic_pointer_cast<const NgTraceSpace<BFT> >( space );
    const Array<int>& map=ng->bemppToNgMap();

    py::array_t<int> ar(map.Size(),map.Addr(0));
    Py_INCREF(ar.ptr());
    return ar.ptr();
}

};


template <typename scal> py::handle eigen_array_cast( Bempp::Matrix<scal>& src)
{
    constexpr ssize_t elem_size = sizeof(scal);
    py::array a;
    if (src.cols()==1)
        a = py::array({ src.size() }, { elem_size * src.innerStride() }, src.data());
    else
        a = py::array({ src.rows(), src.cols() }, { elem_size * src.rowStride(), elem_size * src.colStride() },
                  src.data());

    //if (!writeable)
    //    py::array_proxy(a.ptr())->flags &= ~detail::npy_api::NPY_ARRAY_WRITEABLE_;

    return a.release();
}




py::object pyBemppGridFromNG( std::shared_ptr<ngcomp::MeshAccess> mesh, int domainIndex )
{
    Bempp::Matrix<double> vertices;
    Bempp::Matrix<int> corners;
    std::vector<int> domainIndices;

    NgBem::NgGridFactory::createConnectivityArraysFromNgMesh(mesh, domainIndex,vertices, corners,domainIndices);

    py::module grid = py::module::import( "bempp.api.grid" );

    //return grid.attr( "grid_from_element_data" )(py::cast(vertices),py::cast(corners),py::cast(domainIndices));
    return grid.attr( "grid_from_element_data" )(eigen_array_cast(vertices),eigen_array_cast(corners),py::cast(domainIndices));

//    grid_from_element_data(vertices, elements, domain_indices=[]));
}

py::object pyBemppGridFromNG2( std::shared_ptr<ngcomp::MeshAccess> mesh )
{
    return pyBemppGridFromNG(mesh,-1);
}

/*template<typename T>
void exportNgTraceSpace( const char* classname )
{
    py::implicitly_convertible<std::shared_ptr<Bempp::Grid>,
			       std::shared_ptr<const Bempp::Grid> >();


    typedef NgBem::NgTraceSpace<T> SpaceT;
    py::class_<SpaceT, boost::shared_ptr<SpaceT>,  boost::noncopyable>
      ( classname)
      .def("__init__", (
					    FunctionPointer([]( boost::shared_ptr<const Bempp::Grid> grid, PyWrapper<ngcomp::FESpace> space)
					    {
					      return boost::make_shared<SpaceT>(grid,space.Get());
                        })));
}
*/

PYBIND11_PLUGIN( pyngbem_common )
{
    //exportNgTraceSpace<double>( "RealNgTraceSpace" );
    //exportNgTraceSpace<std::complex<double> >( "ComplexNgTraceSpace" );

    //py::def( "connectivity_arrays_from_ng", &NgBem::NgGridFactory::createConnectivityArraysFromNgMesh);
    py::module module("pyngbem_common");
    module.def( "bempp_grid_from_ng", &pyBemppGridFromNG );
    module.def( "bempp_grid_from_ng", &pyBemppGridFromNG2 );

    py::class_<Bempp::Grid >( module,"My_C_Grid");

    return module.ptr();
}
