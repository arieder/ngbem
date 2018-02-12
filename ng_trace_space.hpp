// Copyright (C) 2011-2012 by the BEM++ Authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ng_trace_space_hpp
#define ng_trace_space_hpp


//#include <common/common.hpp>
//#include <common/types.hpp>
#include <grid/grid_segment.hpp>
#include <grid/grid_view.hpp>

#include <space/scalar_space.hpp>



#include <comp.hpp>
#include <fem.hpp>
#include <ngstd.hpp>


#include <map>
#include <memory>
#include <tbb/mutex.h>


namespace Bempp
{

/** \cond FORWARD_DECL */
class GridView;
template <typename CoordinateType> class BoundingBox;
/** \endcond */

/** \ingroup space
 *  \brief Space of continuous, piecewise polynomial scalar functions. */
};

namespace NgBem{

template <typename BasisFunctionType>
class NgTraceSpace :
    public Bempp::ScalarSpace<BasisFunctionType>
{
public:
    typedef typename Bempp::Space<BasisFunctionType>::CoordinateType CoordinateType;
    typedef typename Bempp::Space<BasisFunctionType>::ComplexType ComplexType;

    typedef typename std::vector<std::pair<unsigned int, Fiber::Shapeset<BasisFunctionType>* > >  ShapesetMap;

    /** \brief Constructor.
     *
     *  Construct a space of continuous functions whose restrictions to
     *  elements of the grid \p grid will be polynomials of order at most \p
     *  polynomialOrder. */
  NgTraceSpace(
      const boost::shared_ptr<const Bempp::Grid>& grid,
      std::shared_ptr<ngcomp::FESpace> fespace);



    /** \brief Constructor.
     *
     *  Construct a space of continuous functions whose restrictions to
     *  elements of the grid \p grid will be polynomials of order at most \p
     *  polynomialOrder. The space will contain only the basis functions deemed
     *  to belong to the segment \p segment; specifically, vertex functions
     *  associated with vertices belonging to \p segment, edge functions
     *  associated with edges belonging to \p segment and bubble function
     *  associated with elements belonging to \p segment. If \p
     *  strictlyOnSegment is \c true, the support of the basis functions is
     *  truncated to the elements that belong to \p segment, too; in this case,
     *  the space may in fact contain discontinuous basis functions when
     *  considered on the whole \p grid, although the basis functions will be
     *  continuous when considered on the chosen grid segment.
     *
     *  An exception is thrown if \p grid is a null pointer.
     */
    NgTraceSpace(
	const boost::shared_ptr<const Bempp::Grid>& grid,
        ngcomp::FESpace* fespace,
            const Bempp::GridSegment& segment,
            bool strictlyOnSegment = false);
    virtual ~NgTraceSpace();

    virtual int domainDimension() const;
    virtual int codomainDimension() const;

    virtual bool isBarycentric() const {
        return false;
    }

    virtual bool spaceIsCompatible(const Bempp::Space<BasisFunctionType>& other) const;

    virtual SpaceIdentifier spaceIdentifier() const {
        return PIECEWISE_POLYNOMIAL_CONTINUOUS_SCALAR;//NG_TRACE_CONTINUOUS_SCALAR;
    }


    /** \brief Return the variant of element \p element.
     *
     *  Possible return values:
     *    - 2: one-dimensional segment,
     *    - 3: triangular element,
     *    - 4: quadrilateral element. */
    virtual Bempp::ElementVariant elementVariant(const Bempp::Entity<0>& element) const;
    virtual void setElementVariant(const Bempp::Entity<0>& element,
                                   Bempp::ElementVariant variant);

    virtual const Fiber::Shapeset<BasisFunctionType>& shapeset(
	const Bempp::Entity<0>& element) const;

    virtual boost::shared_ptr<const Bempp::Space<BasisFunctionType> > discontinuousSpace(
        const boost::shared_ptr<const Bempp::Space<BasisFunctionType> >& self) const;
    virtual bool isDiscontinuous() const;

    virtual size_t globalDofCount() const;
    virtual size_t flatLocalDofCount() const;
    virtual void getGlobalDofs(const Bempp::Entity<0>& element,
			       std::vector<Bempp::GlobalDofIndex>& dofs, std::vector< BasisFunctionType > &  localDofWeights ) const;
  ///HACK!!!!!!
    virtual void getGlobalEdgeDofs(const Bempp::Entity<1>& element,
				   std::vector<Bempp::GlobalDofIndex>& dofs) const;

    virtual void getGlobalVertexDofs(const Bempp::Entity<2>& element,
				     std::vector<Bempp::GlobalDofIndex>& dofs ) const;

    virtual void getGlobalInnerDofs(const Bempp::Entity<0>& element,
				    std::vector<Bempp::GlobalDofIndex>& dofs) const;

  //END HACK


    virtual void global2localDofs(
	const std::vector<Bempp::GlobalDofIndex>& globalDofs,
	std::vector<std::vector<Bempp::LocalDof> >& localDofs,  std::vector<std::vector<BasisFunctionType> >& localDofWeights ) const;
    virtual void flatLocal2localDofs(
	const std::vector<Bempp::FlatLocalDofIndex>& flatLocalDofs,
	std::vector<Bempp::LocalDof>& localDofs) const;

    virtual void getGlobalDofPositions(
	std::vector<Bempp::Point3D<CoordinateType> >& positions) const;
    virtual void getFlatLocalDofPositions(
	std::vector<Bempp::Point3D<CoordinateType> >& positions) const;

    virtual void getGlobalDofBoundingBoxes(
	std::vector<Bempp::BoundingBox<CoordinateType> >& bboxes) const;
    virtual void getFlatLocalDofBoundingBoxes(
	std::vector<Bempp::BoundingBox<CoordinateType> >& bboxes) const;

    virtual void getGlobalDofNormals(
	std::vector<Bempp::Point3D<CoordinateType> >& normals) const;
    virtual void getFlatLocalDofNormals(
	std::vector<Bempp::Point3D<CoordinateType> >& normals) const;

    virtual void dumpClusterIds(
            const char* fileName,
            const std::vector<unsigned int>& clusterIdsOfGlobalDofs) const;
    virtual void dumpClusterIdsEx(
            const char* fileName,
            const std::vector<unsigned int>& clusterIdsOfGlobalDofs,
            Bempp::DofType dofType) const;


    const ngstd::Array<int>& ngToBemppMap() const;
    const ngstd::Array<int>& bemppToNgMap() const;

    inline int FESpaceDofCount() const {return m_fespace->GetNDof();}

    /*
    inline void setLocalHeapForThread(int threadId, ngcomp::LocalHeap* lh )
    {
        m_lh[threadId]=lh;
        m_lastLHVal[threadId]=lh->GetPointer();
        //updateShapesets( threadId );
    }


    inline ngcomp::LocalHeap* getLocalHeapForThread( int threadId ) const
    {
        return m_lh[threadId];
    }

    inline void setAllLocalHeaps( ngcomp::LocalHeap* lh )
    {
        for ( int i=0;i<m_lh.size();i++ )
            setLocalHeapForThread( i, lh );
    }*/

private:
    void initialize();
    void assignDofsImpl();
    void updateNgMaps();

    void updateShapesets();

    inline Fiber::Shapeset<BasisFunctionType>* findShapeset( int orientation ) const
    {
        typedef std::pair<unsigned int, Fiber::Shapeset<BasisFunctionType>* > ShapesetPair;
        for ( const ShapesetPair& sp : m_shapesets )
        {
            if ( sp.first == orientation )
            {
                return sp.second;
            }
        }
        return 0;
    }

    inline int computeOrientationFlag( const Bempp::Entity<0>& element ) const
    {
        auto ma=m_fespace->GetMeshAccess();
        const Bempp::IndexSet& iset = m_view->indexSet();
        const unsigned int edgeToVertexMap[][2]={{0, 1}, {0, 2}, {1, 2}};;//{{0, 1}, {0, 2}, {1, 2}};

        Bempp::EntityIndex elementId = iset.entityIndex(element);


        //for ( int i=0;i<3;i++ )
        //    vertexIndices[i]=iset.subEntityIndex( element,  i,  2 );
        //vertexIndices[0] = iset.subEntityIndex( element,  1,  2 );
        //vertexIndices[1] = iset.subEntityIndex( element,  2,  2 );
        //vertexIndices[2] = iset.subEntityIndex( element,  0,  2 );
        ngbla::Array< int> vnums;
        ma->GetSElVertices ( elementId,   vnums );

        int orientation=0; //will encode the orientations of the edges to determine which shapeset to use
        for ( int edge=0;edge<3;edge++ )
        {
            ngstd::INT<2> edgeV=ngfem::ET_trait<ngfem::ET_TRIG>::GetEdgeSort( edge,vnums );
            //const int edgeOrientation=( vertexIndices[edgeToVertexMap[edge][0]]< vertexIndices[edgeToVertexMap[edge][1]] ) ? 1:0;
            //const int edgeOrientation=( vertexIndices[edgeV[0]] < vertexIndices[edgeV[1]]  ) ? 0:1; //check if the sort swapped something
            //std::cout<<"edge : "<<edgeOrientation<<std::endl;
            //orientation|=edgeOrientation;
            orientation+=edgeV[0];
            orientation+=edgeV[1]*3;
            orientation*=9;
            //orientation<<=1;
        }

        //std::cout<<"orientation: "<<orientation<<std::endl;

        //orientation<<=4;
        ngstd::INT<4> f = ngfem::ET_trait<ngfem::ET_TRIG>::GetFaceSort ( 0,  vnums);

        //std::cout<<"f: "<<f<<std::endl;
        char faceOrientation=0;
        int w=1;
        for ( int i=0;i<3;i++ )
        {
            faceOrientation+=f[i];
            w*=3;
        }
        //std::cout<<"orientation "<<( unsigned int ) orientation<<std::endl;;
        //std::cout<<"faceOrient: "<<( unsigned int )faceOrientation<<std::endl;
        orientation+=faceOrientation;

        return orientation;


    }


private:
    /** \cond PRIVATE */
    int m_polynomialOrder;
    shared_ptr<ngcomp::FESpace> m_fespace;
    //std::vector<ngcomp::LocalHeap* > m_lh;
    ngstd::LocalHeap* m_lh;
    ngstd::LocalHeap* m_spaceHeap;
    std::vector<void*> m_lastLHVal;
    ShapesetMap  m_shapesets;
    Bempp::GridSegment m_segment;
    bool m_strictlyOnSegment;
    boost::scoped_ptr<Fiber::Shapeset<BasisFunctionType> > m_triangleShapeset;
    int m_triangleOrientation[3];
    std::unique_ptr<Bempp::GridView> m_view;
    std::vector<std::vector<Bempp::GlobalDofIndex> > m_local2globalDofs;
    std::vector<std::vector<BasisFunctionType> > m_local2globalDofWeights;
    std::vector<std::vector<Bempp::LocalDof> > m_global2localDofs;
    std::vector<Bempp::LocalDof> m_flatLocal2localDofs;
    size_t m_flatLocalDofCount;
    std::vector<Bempp::BoundingBox<CoordinateType> > m_globalDofBoundingBoxes;
    mutable boost::shared_ptr<Bempp::Space<BasisFunctionType> > m_discontinuousSpace;
    mutable tbb::mutex m_discontinuousSpaceMutex;
    /** \endcond */

  ///HACK
    std::vector<Bempp::GlobalDofIndex> m_vertexGlobalDofs;
    std::vector<Bempp::GlobalDofIndex> m_edgeStartingGlobalDofs;
    std::vector<Bempp::GlobalDofIndex> m_bubbleStartingGlobalDofs;

    ngstd::Array<int> m_ngToBempp;
    ngstd::Array<int> m_bemppToNg;

};


}
#endif
