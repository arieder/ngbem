#ifndef ng_trace_shapeset_h
#define ng_trace_shapeset_h

#include <common/common.hpp>

#include "fiber/basis.hpp"

#include "fiber/basis_data.hpp"
#include "fiber/dune_basis_helper.hpp"

#include <dune/localfunctions/lagrange/pk2d/pk2dlocalbasis.hh>


using namespace Fiber;

#include <comp.hpp>

/***
HACKISH!!

 **/

// general case
template <typename T>
struct RealPart {
    typedef T type;
};

// std::complex
template <typename T>
struct RealPart<std::complex<T> > {
    typedef T type;
};


/** \brief Shapeset composed a ng_trace_ basis of polynomials up to a specified order. */
template <typename ValueType>
class NgTraceShapeset : public Fiber::Basis<ValueType>
{

public:
    typedef typename Basis<ValueType>::CoordinateType CoordinateType;
    enum {DimDomain = 2, DimRange = 1};
private:
    int m_id;
    int m_size;
    const ngfem::ScalarFiniteElement<DimDomain>& m_fel;
    std::shared_ptr<ngcomp::FESpace> m_ngspace;
    ngcomp::LocalHeap& m_lh;
    int m_polynomialOrder;

public:
    NgTraceShapeset( int id, std::shared_ptr<ngcomp::FESpace> ngspace, ngcomp::LocalHeap& lh):
        m_id( id ), m_ngspace( ngspace ), m_lh( lh ),m_fel( dynamic_cast<const ngfem::ScalarFiniteElement<DimDomain>&> ( ngspace->GetSFE( id, lh ) ) )
    {
        if ( ngspace->GetMeshAccess()->GetNSE() <= id )
        {
            std::cout<<"invalid"<<std::endl;
            throw ngstd::Exception( "NgTraceShapeset::The fespace has not enough surface elements" );
        }

        m_size=m_fel.GetNDof();
        //std::cout<<"ngspace: "<<ngspace->GetClassName()<<" "<<size()<<std::endl;
        m_polynomialOrder=m_fel.Order();
    }

    ~NgTraceShapeset()
    {
        //std::cout<<"shapeset is dying!"<<std::endl;
    }

    virtual int size() const {
	return m_size;
    }

    virtual int order() const {
	return m_polynomialOrder;
    }

    virtual void evaluate(size_t what,
			  const Bempp::Matrix<CoordinateType>& points,
			  LocalDofIndex localDofIndex,
			  BasisData<ValueType>& data) const {
        //std::cout<<"eval"<<std::endl;
#ifndef NDEBUG
	if (localDofIndex != ALL_DOFS &&
		(localDofIndex < 0 || size() <= localDofIndex))
	    throw std::invalid_argument("MyShapeset::"
					"evaluate(): Invalid localDofIndex");

        if (m_ngspace==0|| m_ngspace->GetMeshAccess()->GetNSE() <= 0 )
        {
            throw ngcomp::Exception( "no surface elements available in current ng fespace" );
        }
#endif

        /*
        ngla::Mat<DimDomain, DimDomain> trafoA;
        trafoA( 0, 0 )=-1.0; trafoA( 0, 1 )=-1.0;
        trafoA( 1, 0 )=1.0; trafoA( 1, 1 ) =0;

        ngla::Vec<DimDomain> trafoB;
        trafoB( 0 )=1.0; trafoB( 1 )=0;

        */

        //Bempp::Vector<CoordinateType> apoint;
        ngla::Vec<DimDomain> point;
	if (what & VALUES)
	{
            //std::cout<<"values"<<std::endl;
	    const int functionCount = (localDofIndex == ALL_DOFS ) ? size() : 1;
	    const int pointCount = points.cols();
	    data.values.set_size(DimRange, functionCount, pointCount);

            ngbla::Vector<double> values( m_fel.GetNDof() );

            for (int pointIndex = 0; pointIndex < pointCount; ++pointIndex)
	    {
                //apoint=points.col( pointIndex );
                //ngla::FlatVector<CoordinateType> tmp( apoint.size(), ( CoordinateType* ) apoint.memptr() );
                point[0]=-points( 0, pointIndex )-points( 1, pointIndex ) +1.0;
                point[1]=points( 0, pointIndex );

                ngfem::IntegrationPoint ip ( point,1 );
                //ip.SetNr( pointIndex ); //is this needed/useful/safe?

                m_fel.CalcShape(ip, values);

                //for ( int i=0;i<m_fel.GetNDof(); i++ )
                //ordered_values[convertNgToBemppIndices( i,m_polynomialOrder )]=values[i];

                //std::cout<<"done"<<std::endl;
                //std::cout<<"orig: "<<values<<std::endl;
                //std::cout<<"ordered: "<<ordered_values<<std::endl;

		if (localDofIndex == ALL_DOFS)
		    for (int functionIndex = 0; functionIndex < functionCount; ++functionIndex)
			data.values(0, functionIndex, pointIndex) = values[functionIndex];
		else
		    data.values(0, 0, pointIndex) = values[localDofIndex];
	    }

	}

	if (what & DERIVATIVES)
	{
	    assert(localDofIndex == ALL_DOFS ||
		   (localDofIndex >= 0 && localDofIndex < size()));

	    const int functionCount = localDofIndex == ALL_DOFS ? size() : 1;
	    const int pointCount = points.cols();

            ngbla::MatrixFixWidth<DimDomain> dshape( m_fel.GetNDof());
	    data.derivatives.set_size(1, DimDomain, functionCount, pointCount);
            Bempp::Vector<CoordinateType> tmpVec( functionCount );

            Bempp::Vector<CoordinateType> diff( (int) DimDomain );
	    for (int pointIndex = 0; pointIndex < pointCount; ++pointIndex)
	    {
                point[0]=-points( 0, pointIndex )-points( 1, pointIndex ) +1.0;
                point[1]=points( 0, pointIndex );


                ngfem::IntegrationPoint ip(point,1.0);
                ip.SetNr( pointIndex ); //is this needed/useful/safe?

		m_fel.CalcDShape( ip, dshape );
		if (localDofIndex == ALL_DOFS)
                {
		    for (int functionIndex = 0; functionIndex < functionCount; ++functionIndex)
                    {
                        data.derivatives(0, 0, functionIndex, pointIndex )=-dshape( functionIndex, 0 )+dshape(functionIndex, 1 );
                        data.derivatives(0, 1, functionIndex, pointIndex )=-dshape( functionIndex, 0 );
                    }
                }
		else
                {
                    data.derivatives(0, 0, 0, pointIndex )=-dshape( localDofIndex, 0 )+dshape( localDofIndex, 1 );
                    data.derivatives(0, 1, 0, pointIndex )=-dshape( localDofIndex, 0 );
                }

	    }
	}
    }

    void evaluateFunctions( const Bempp::Vector<CoordinateType>& point, Bempp::Vector<ValueType>& ordered_values )
    {
#ifndef NDEBUG
        if (m_ngspace==0|| m_ngspace->GetMeshAccess()->GetNSE() <= 0 )
        {
            throw ngcomp::Exception( "no surface elements available in current ng fespace" );
        }
#endif
        const ngfem::ScalarFiniteElement<DimDomain>& fel=dynamic_cast<const ngfem::ScalarFiniteElement<DimDomain>&> ( m_ngspace->GetSFE( 0 ,m_lh ) ) ;

        ngbla::Vector<ValueType> values( fel.GetNDof() );

        ngla::Mat<DimDomain, DimDomain> trafoA;
        trafoA( 0, 0 )=-1.0; trafoA( 0, 1 )=-1.0;
        trafoA( 1, 0 )=1.0; trafoA( 1, 1 ) =0;

        ngla::Vec<DimDomain> trafoB;
        trafoB( 0 )=1.0; trafoB( 1 )=0;


        ngla::Vec<DimDomain> npoint;
        //std::cout<<"values"<<std::endl;

        ngla::FlatVector<CoordinateType> tmp( point.size(), ( CoordinateType* ) point.memptr() );
        npoint=trafoA*tmp+trafoB;

        ngfem::IntegrationPoint ip ( npoint,1 );

        m_fel.CalcShape(ip, values);


        for ( int i=0;i<fel.GetNDof(); i++ )
            ordered_values[i]=values[i];
        //std::cout<<"done"<<std::endl;
        /*std::cout<<"orig: "<<values<<std::endl;
        std::cout<<"ordered: "<<ordered_values<<std::endl;
        std::cout<<"done"<<std::endl;*/

    }


    void evaluateFunctionsD( const Bempp::Vector<CoordinateType>& point, Bempp::Matrix<ValueType>& values )
    {
#ifndef NDEBUG
        if (m_ngspace==0|| m_ngspace->GetMeshAccess()->GetNSE() <= 0 )
        {
            throw ngcomp::Exception( "no surface elements available in current ng fespace" );
        }
#endif
        const ngfem::ScalarFiniteElement<DimDomain>& fel=dynamic_cast<const ngfem::ScalarFiniteElement<DimDomain>&> ( m_ngspace->GetSFE( 0 ,m_lh ) ) ;

        ngbla::FlatMatrixFixWidth<DimDomain> dshape( m_fel.GetNDof(), m_lh );

        ngla::Mat<DimDomain, DimDomain> trafoA;
        trafoA( 0, 0 )=-1.0; trafoA( 0, 1 )=-1.0;
        trafoA( 1, 0 )=1.0; trafoA( 1, 1 ) =0;

        ngla::Vec<DimDomain> trafoB;
        trafoB( 0 )=1.0; trafoB( 1 )=0;


        ngla::Vec<DimDomain> npoint;
        //std::cout<<"values"<<std::endl;

        ngla::FlatVector<CoordinateType> tmp( point.size(), ( CoordinateType* ) point.memptr() );
        npoint=trafoA*tmp+trafoB;

        ngfem::IntegrationPoint ip ( npoint,1 );

        m_fel.CalcDShape(ip, dshape);

        for ( int i=0;i<fel.GetNDof(); i++ )
        {
            const ngla::Vec<DimDomain> ds=dshape.Row( i );
            const ngla::Vec<DimDomain> t_ds;

            values( i, 0 )=-ds( 0 )+ds( 1 );
            values( i, 1 )=-ds( 0 );
        }
    }



    static inline unsigned int functionCount(unsigned int order)
    {
	return (order+1)*(order+2)/2 ;
    }

    static inline unsigned int dofCountPerEdge(unsigned int order)
    {
	return order-1;
    }

    static inline unsigned int interiorDofCount(unsigned int order)
    {
	return order-2;
    }


    static inline int convertNgToBemppIndices( int i, int order )
    {
        return i;
    }


    static inline int convertNgToBemppIndicesIgnoreVertexOrder( int i, int order)
    {
        return i;

    }



};



#endif

