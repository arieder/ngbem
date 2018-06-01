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

#include "ng_trace_space.hpp"


#include <space/space_helper.hpp>

#include <assembly/discrete_sparse_boundary_operator.hpp>
#include <common/acc.hpp>
#include <common/boost_make_shared_fwd.hpp>
#include <common/bounding_box_helpers.hpp>
#include <fiber/explicit_instantiation.hpp>


#include "ng_trace_shapeset.hpp"
//#include "ng_trace_discontinuous_space.hpp"
#include "space/piecewise_polynomial_discontinuous_scalar_space.hpp"


#include <grid/entity.hpp>
#include <grid/entity_iterator.hpp>
#include <grid/geometry.hpp>
#include <grid/grid.hpp>
#include <grid/grid_segment.hpp>
#include <grid/grid_view.hpp>
#include <grid/mapper.hpp>
#include <grid/reverse_element_mapper.hpp>
#include <grid/vtk_writer.hpp>
#include <grid/index_set.hpp>
#include <grid/id_set.hpp>

#include <stdexcept>
#include <iostream>

#include <boost/array.hpp>

using namespace NgBem;
using namespace Bempp;
using namespace boost;
using namespace ngcomp;

template <typename BasisFunctionType>
NgTraceSpace<BasisFunctionType>::
NgTraceSpace(const boost::shared_ptr<const Grid>& grid,
             std::shared_ptr<FESpace> fespace) :
    ScalarSpace<BasisFunctionType>(grid), m_fespace( fespace )  ,
    m_flatLocalDofCount(0), m_segment(GridSegment::wholeGrid(*grid)),
    m_strictlyOnSegment(false)
{

    initialize();


}

template <typename BasisFunctionType>
bool NgTraceSpace<BasisFunctionType>::
spaceIsCompatible(const Space<BasisFunctionType> &other) const
{

       typedef NgTraceSpace<BasisFunctionType> thisSpaceType;

       if (other.grid().get()!=this->grid().get()) return false;

       if (other.spaceIdentifier()==this->spaceIdentifier()){
           // Try to typecast the other space down.
           const thisSpaceType& temp = dynamic_cast<const thisSpaceType&>(other);
           if (this->m_polynomialOrder==temp.m_polynomialOrder)
               return true;
           else
               return false;
       }
       else
           return false;
}


template <typename BasisFunctionType>
NgTraceSpace<BasisFunctionType>::
NgTraceSpace(const boost::shared_ptr<const Grid>& grid,
                                         FESpace* fespace,
                                         const GridSegment& segment,
                                         bool strictlyOnSegment) :
    ScalarSpace<BasisFunctionType>(grid), m_fespace( fespace ),
    m_flatLocalDofCount(0), m_segment(segment),
    m_strictlyOnSegment(strictlyOnSegment)
{
    initialize();

}

template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::initialize()
{
    const int gridDim = this->grid()->dim();
    if (gridDim != 2)
        throw std::invalid_argument("NgTraceSpace::"
                                    "NgTraceSpace(): "
                                    "2-dimensional grids are supported");
    m_view = this->grid()->leafView();
    m_spaceHeap=new ngstd::LocalHeap(5000, "ngspaceheap");
    m_polynomialOrder=m_fespace->GetOrder();
    //m_triangleShapeset.reset(new Fiber::LagrangeScalarShapeset<3, BasisFunctionType, 2>());

    //m_triangleShapeset.reset(new Fiber::MyShapeset<3, BasisFunctionType, 2>());


    if ( m_polynomialOrder <=3 )
    {
        m_triangleShapeset.reset(new NgTraceShapeset<BasisFunctionType>(0, m_fespace, *m_spaceHeap ) );
        const unsigned int edgeToVertexMap[][2]= { { 2,  0 },
                                                   { 1,  2 },
                                                   { 0,  1 } };

        const Bempp::EntityIndex elementId = 0;

        auto ma=m_fespace->GetMeshAccess();
        ngbla::Array< int> vnums;
        ma->GetSElVertices ( elementId,   vnums );

        for ( int edge=0;edge<3;edge++ )
        {
            int orientation= ( vnums[edgeToVertexMap[edge][0]] > vnums[edgeToVertexMap[edge][1]]  ) ? -1:1;
            m_triangleOrientation[edge]=orientation;
        }
    }else
    {
        m_shapesets.reserve( 6 );
        updateShapesets();
    }
    assignDofsImpl();

    updateNgMaps();

    //m_shapesets.resize( MAX_NUM_THREADS );
}

template <typename BasisFunctionType>
NgTraceSpace<BasisFunctionType>::
~NgTraceSpace()
{
    std::cout<<"ending ng trace space"<<std::endl;
    delete m_spaceHeap;
}

template <typename BasisFunctionType>
int NgTraceSpace<BasisFunctionType>::domainDimension() const
{
    return this->grid()->dim();
}

template <typename BasisFunctionType>
int NgTraceSpace<BasisFunctionType>::codomainDimension() const
{
    return 1;
}

template <typename BasisFunctionType>
const Fiber::Shapeset<BasisFunctionType>&
NgTraceSpace<BasisFunctionType>::shapeset(
        const Entity<0>& element) const
{
    if (elementVariant(element) == 3)
    {
        const IndexSet& iset = m_view->indexSet();
        const EntityIndex id=iset.entityIndex( element );
        Fiber::Shapeset<BasisFunctionType>* shapeset;
        if ( m_polynomialOrder<=3)
        {
            return *m_triangleShapeset;
        }



        int orientation=computeOrientationFlag( element ); //encodes the orientations of the edges and triangle itself to determine which shapeset to use
        //std::cout<<"accessing "<<orientation<<std::endl;

        shapeset=findShapeset( orientation );
        return *shapeset;
    }
    throw std::logic_error("NgTraceSpace::shapeset(): "
                           "invalid element variant, this shouldn't happen!");
}

template <typename BasisFunctionType>
ElementVariant NgTraceSpace<BasisFunctionType>::elementVariant(
        const Entity<0>& element) const
{
    GeometryType type = element.type();
    if (type.isLine())
        return 2;
    else if (type.isTriangle())
        return 3;
    else if (type.isQuadrilateral())
        return 4;
    else
        throw std::runtime_error("NgTraceSpace::"
                                 "elementVariant(): invalid geometry type, "
                                 "this shouldn't happen!");
}

template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::setElementVariant(
        const Entity<0>& element, ElementVariant variant)
{
    if (variant != elementVariant(element))
        // for this space, the element variants are unmodifiable,
        throw std::runtime_error("NgTraceSpace::"
                                 "setElementVariant(): invalid variant");
}

template <typename BasisFunctionType>
boost::shared_ptr<const Space<BasisFunctionType> >
NgTraceSpace<BasisFunctionType>::discontinuousSpace(
    const boost::shared_ptr<const Space<BasisFunctionType> >& self) const
{
    //throw std::runtime_error( "discontinuousSpace is not implemented yet" );
   if (!m_discontinuousSpace) {
       std::cout<<"requesting a DG space. hope for the best!"<<std::endl;
       tbb::mutex::scoped_lock lock(m_discontinuousSpaceMutex);
//       typedef NgTraceDiscontinuousScalarSpace<BasisFunctionType>
//         DiscontinuousSpace;
       typedef PiecewisePolynomialDiscontinuousScalarSpace<BasisFunctionType>
           DiscontinuousSpace;


       if (!m_discontinuousSpace)
           m_discontinuousSpace.reset(new DiscontinuousSpace(
                                          this->grid(), m_polynomialOrder,
                                          m_segment));
   }
   return m_discontinuousSpace;
}

template <typename BasisFunctionType>
bool
NgTraceSpace<BasisFunctionType>::isDiscontinuous() const
{
    return false;
}

template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::assignDofsImpl()
{
    // TODO: refactor this function, it's way too long!

    // In addition to DOF assignment, this function also precalculates bounding
    // boxes of global DOFs

    const int elementCount = m_view->entityCount(0);
    if (elementCount == 0)
        return;
    const int gridDim = this->domainDimension();
    if (gridDim != 2)
        throw std::runtime_error("NgTraceSpace::"
                                 "assignDofsImpl(): only 2-dimensional grids "
                                 "are supported at present");
    const int vertexCodim = gridDim;
    const int edgeCodim = vertexCodim - 1;

    // const Mapper& elementMapper = m_view->elementMapper();
    const IndexSet& indexSet = m_view->indexSet();

    // Map vertices to global dofs
    const int vertexCount = m_view->entityCount(2);
    // At first, the elements of this vector will be set to the number of
    // DOFs corresponding to a given vertex or to -1 if that vertex is to be
    // ignored
    std::vector<GlobalDofIndex> vertexGlobalDofs(vertexCount);
    for (int i = 0; i < vertexCount; ++i)
        if (m_segment.contains(gridDim, i))
            acc(vertexGlobalDofs, (size_t)i) = 1;
        else
            acc(vertexGlobalDofs, (size_t)i) = -1;

    // Map edges to global dofs
    const int edgeCount = m_view->entityCount(1);
    const int internalDofCountPerEdge = m_polynomialOrder - 1;
    // At first, the elements of this vector will be set to the number of
    // DOFs corresponding to a given edge or to -1 if that edge is to be
    // ignored
    std::vector<GlobalDofIndex> edgeStartingGlobalDofs(edgeCount);
    for (int i = 0; i < edgeCount; ++i)
        if (m_segment.contains(gridDim - 1, i))
            acc(edgeStartingGlobalDofs, i) = internalDofCountPerEdge;
        else
            acc(edgeStartingGlobalDofs, i) = -1;

    // Map element interiors to global dofs
    // and, if striclyOnSegment is set, detect vertices and edges not belonging
    // to any element on segment
    const int bubbleDofCountPerTriangle =
        std::max(0, (m_polynomialOrder - 1) * (m_polynomialOrder - 2) / 2);
    const int bubbleDofCountPerQuad =
        std::max(0, (m_polynomialOrder - 1) * (m_polynomialOrder - 1));
    // At first, the elements of this vector will be set to the number of
    // DOFs corresponding to a given element or to -1 if that element is to be
    // ignored
    std::vector<GlobalDofIndex> bubbleStartingGlobalDofs(elementCount);
    std::vector<bool> noElementAdjacentToVertexIsOnSegment(vertexCount, true);
    std::vector<bool> noElementAdjacentToEdgeIsOnSegment(edgeCount, true);
    std::unique_ptr<EntityIterator<0> > it = m_view->entityIterator<0>();
    while (!it->finished()) {
        const Entity<0>& element = it->entity();
        EntityIndex elementIndex = indexSet.entityIndex(element);
        int vertexCount = element.template subEntityCount<2>();
        if (vertexCount != 3 && vertexCount != 4)
            throw std::runtime_error("NgTraceSpace::"
                                     "assignDofsImpl(): elements must be "
                                     "triangular or quadrilateral");
        if (m_segment.contains(0, elementIndex)) {
            acc(bubbleStartingGlobalDofs, elementIndex) =
                vertexCount == 3 ? bubbleDofCountPerTriangle : bubbleDofCountPerQuad;
            if (m_strictlyOnSegment)
                for (int i = 0; i < vertexCount; ++i) {
                    int index = indexSet.subEntityIndex(element, i, gridDim);
                    acc(noElementAdjacentToVertexIsOnSegment, index) = false;
                    index = indexSet.subEntityIndex(element, i, gridDim - 1);
                    acc(noElementAdjacentToEdgeIsOnSegment, index) = false;
                }
        }
        else
            acc(bubbleStartingGlobalDofs, elementIndex) = -1;
        it->next();
    }

    // If strictlyOnSegment is set, deactivate vertices and edges not adjacent
    // to any element in segment
    if (m_strictlyOnSegment) {
        for (int i = 0; i < vertexCount; ++i)
            if (acc(noElementAdjacentToVertexIsOnSegment, i))
                acc(vertexGlobalDofs, i) = -1;

        for (int i = 0; i < edgeCount; ++i)
            if (acc(noElementAdjacentToEdgeIsOnSegment, i))
                acc(edgeStartingGlobalDofs, i) = -1;
    }

    // Assign global dofs to entities
    int globalDofCount_ = 0;
    for (int i = 0; i < vertexCount; ++i)
        if (acc(vertexGlobalDofs, i) == 1)
            acc(vertexGlobalDofs, i) = globalDofCount_++;
    for (int i = 0; i < edgeCount; ++i) {
        int dofCount = acc(edgeStartingGlobalDofs, i);
        if (dofCount > 0) {
            acc(edgeStartingGlobalDofs, i) = globalDofCount_;
            globalDofCount_ += dofCount;
        }
    }
    for (int i = 0; i < elementCount; ++i) {
        int dofCount = acc(bubbleStartingGlobalDofs, i);
        if (dofCount > 0) {
            acc(bubbleStartingGlobalDofs, i) = globalDofCount_;
            globalDofCount_ += dofCount;
        }
    }

    // Initialise DOF maps
    const int localDofCountPerTriangle =
        (m_polynomialOrder + 1) * (m_polynomialOrder + 2) / 2;
    const int localDofCountPerQuad =
        (m_polynomialOrder + 1) * (m_polynomialOrder + 1);
    m_local2globalDofs.clear();
    std::vector<GlobalDofIndex> prototypeGlobalDofs;
    prototypeGlobalDofs.reserve(localDofCountPerTriangle);
    m_local2globalDofs.resize(elementCount, prototypeGlobalDofs);
    m_global2localDofs.clear();
    // std::vector<LocalDof> prototypeLocalDofs;
    // prototypeLocalDofs.reserve(localDofCountPerTriangle);
    m_global2localDofs.resize(globalDofCount_/*, prototypeLocalDofs*/);


    m_local2globalDofWeights.clear();
    m_local2globalDofWeights.resize( elementCount );


     //we can use the same shapeset for all elements,
    //to save some time/storage. But to do so we need to
    //properly compute weights based on edge orientation
    const int functionCount=localDofCountPerTriangle;//;m_triangleShapeset->size();

    auto ma=m_fespace->GetMeshAccess();
    ngbla::Array< int> vnums;

    const unsigned int edgeToVertexMap[][2]= { { 2,  0 },
                                               { 1,  2 },
                                               { 0,  1 } };


    // Initialise bounding-box caches
    Bempp::BoundingBox<CoordinateType> model;
    model.lbound.x = std::numeric_limits<CoordinateType>::max();
    model.lbound.y = std::numeric_limits<CoordinateType>::max();
    model.lbound.z = std::numeric_limits<CoordinateType>::max();
    model.ubound.x = -std::numeric_limits<CoordinateType>::max();
    model.ubound.y = -std::numeric_limits<CoordinateType>::max();
    model.ubound.z = -std::numeric_limits<CoordinateType>::max();
    m_globalDofBoundingBoxes.resize(globalDofCount_, model);


    // Iterate over elements
    it = m_view->entityIterator<0>();
    Bempp::Matrix<CoordinateType> vertices;
    Bempp::Vector<CoordinateType> dofPosition;
    m_flatLocalDofCount = 0;
    std::vector<int> gdofAccessCounts(globalDofCount_, 0);
    while (!it->finished()) {
        const Entity<0>& element = it->entity();
        const Geometry& geo = element.geometry();
        EntityIndex elementIndex = indexSet.entityIndex(element);
        bool elementContained = !m_strictlyOnSegment ||
                    m_segment.contains(0, elementIndex);

        geo.getCorners(vertices);
        int vertexCount = vertices.cols();

        ma->GetSElVertices ( elementIndex,   vnums );


        // List of global DOF indices corresponding to the local DOFs of the
        // current element
        std::vector<GlobalDofIndex>& globalDofs = acc(m_local2globalDofs, elementIndex);
        std::vector<BasisFunctionType>& globalDofWeights= acc(  m_local2globalDofWeights,   elementIndex );


        if (vertexCount == 3) {
            std::vector<int> ldofAccessCounts(localDofCountPerTriangle, 0);
            boost::array<int, 3> vertexIndices;
            for (int i = 0; i < 3; ++i)
                acc(vertexIndices, i) = indexSet.subEntityIndex(element, i, vertexCodim);
            globalDofs.resize(localDofCountPerTriangle);
            globalDofWeights.resize(  localDofCountPerTriangle );

            int ldof=0;
            // vertex dofs
            {
                int gdof;

                ldof = 0;
                acc( globalDofWeights, ldof )=1.0;

                if (elementContained)
                    gdof = vertexGlobalDofs[acc(vertexIndices, 0)];
                else
                    gdof = -1;
                if (gdof >= 0) {
                    acc(globalDofs, ldof) = gdof;

                    acc(m_global2localDofs, gdof).push_back(LocalDof(elementIndex, ldof));
                    ++acc(gdofAccessCounts, gdof);
                    extendBoundingBox(acc(m_globalDofBoundingBoxes, gdof), vertices);
                    setBoundingBoxReference<CoordinateType>(
                        acc(m_globalDofBoundingBoxes, gdof), vertices.col(0));
                    ++m_flatLocalDofCount;
                } else
                    acc(globalDofs, ldof) = -1;
                ++acc(ldofAccessCounts, ldof);

                ldof++;
                acc( globalDofWeights, ldof )=1.0;

                if (elementContained)
                    gdof = vertexGlobalDofs[acc(vertexIndices, 1)];
                else
                    gdof = -1;
                if (gdof >= 0) {
                    acc(globalDofs, ldof) = gdof;
                    acc(m_global2localDofs, gdof).push_back(LocalDof(elementIndex, ldof));
                    ++acc(gdofAccessCounts, gdof);
                    extendBoundingBox(acc(m_globalDofBoundingBoxes, gdof), vertices);
                    setBoundingBoxReference<CoordinateType>(
                        acc(m_globalDofBoundingBoxes, gdof), vertices.col(1));
                    ++m_flatLocalDofCount;
                } else
                    acc(globalDofs, ldof) = -1;
                ++acc(ldofAccessCounts, ldof);

                ldof++;
                acc( globalDofWeights, ldof )=1.0;

                if (elementContained)
                    gdof = vertexGlobalDofs[acc(vertexIndices, 2)];
                else
                    gdof = -1;
                if (gdof >= 0) {
                    acc(globalDofs, ldof) = gdof;
                    acc(m_global2localDofs, gdof).push_back(LocalDof(elementIndex, ldof));
                    ++acc(gdofAccessCounts, gdof);
                    extendBoundingBox(acc(m_globalDofBoundingBoxes, gdof), vertices);
                    setBoundingBoxReference<CoordinateType>(
                        acc(m_globalDofBoundingBoxes, gdof), vertices.col(2));
                    ++m_flatLocalDofCount;
                } else
                    acc(globalDofs, ldof) = -1;
                ++acc(ldofAccessCounts, ldof);
            }

            // edge dofs
            if (m_polynomialOrder >= 2) {
                int start, end, step;
                int edgeIndex;

                ///ngsolve orders the edges differently. they correspond to 1,2,0
                edgeIndex = indexSet.subEntityIndex(element, 1, edgeCodim);

                int edge=1;
                int orientation= ( vnums[edgeToVertexMap[edge][0]] > vnums[edgeToVertexMap[edge][1]]  ) ? -1:1;
                orientation*=m_triangleOrientation[edge];
                //std::cout<<"orientation: "<<edge<<" : "<<orientation<<" base: "<<m_triangleOrientation[edge]<<std::endl;
                //const int start=m_edgeStartingGlobalDofs[index];


                dofPosition = 0.5 * (vertices.col(0) + vertices.col(2));
                if (acc(edgeStartingGlobalDofs, edgeIndex) >= 0 &&
                        elementContained) {
                        start = acc(edgeStartingGlobalDofs, edgeIndex);
                        end = start + internalDofCountPerEdge;
                        step = 1;
                    for (int ldofy = 1, gdof = start; gdof != end; ++ldofy, gdof += step) {
                        ldof++;

                        if ( ldofy % 2 == 1 )
                            acc( globalDofWeights, ldof )=1.0;
                        else
                            acc( globalDofWeights, ldof )=orientation;;


                        acc(globalDofs, ldof) = gdof;
                        acc(m_global2localDofs, gdof).push_back(LocalDof(elementIndex, ldof));
                        ++acc(ldofAccessCounts, ldof);
                        ++acc(gdofAccessCounts, gdof);
                        extendBoundingBox(acc(m_globalDofBoundingBoxes, gdof), vertices);
                        setBoundingBoxReference<CoordinateType>(
                            acc(m_globalDofBoundingBoxes, gdof), dofPosition);
                        ++m_flatLocalDofCount;
                    }
                } else
                    for (int ldofy = 1; ldofy <= internalDofCountPerEdge; ++ldofy) {
                        ldof++;
                        acc(globalDofs, ldof) = -1;

                        if ( ldofy % 2 == 1 )
                            acc( globalDofWeights, ldof )=1.0;
                        else
                            acc( globalDofWeights, ldof )=orientation;

                        ++acc(ldofAccessCounts, ldof);
                    }

                edgeIndex = indexSet.subEntityIndex(element, 2, edgeCodim);

                edge=2;
                orientation= ( vnums[edgeToVertexMap[edge][0]] > vnums[edgeToVertexMap[edge][1]]  ) ? -1:1;
                orientation*=m_triangleOrientation[edge];

                dofPosition = 0.5 * (vertices.col(1) + vertices.col(2));
                if (acc(edgeStartingGlobalDofs, edgeIndex) >= 0 &&
                        elementContained) {
                        start = acc(edgeStartingGlobalDofs, edgeIndex);
                        end = start + internalDofCountPerEdge;
                        step = 1;
                    for (int ldofy = 1, gdof = start; gdof != end; ++ldofy, gdof += step) {
                        ldof++;
                        if ( ldofy % 2 == 1 )
                            acc( globalDofWeights, ldof )=1.0;
                        else
                            acc( globalDofWeights, ldof )=orientation;

                        acc(globalDofs, ldof) = gdof;
                        acc(m_global2localDofs, gdof).push_back(LocalDof(elementIndex, ldof));
                        ++acc(ldofAccessCounts, ldof);
                        ++acc(gdofAccessCounts, gdof);
                        extendBoundingBox(acc(m_globalDofBoundingBoxes, gdof), vertices);
                        setBoundingBoxReference<CoordinateType>(
                            acc(m_globalDofBoundingBoxes, gdof), dofPosition);
                        ++m_flatLocalDofCount;
                    }
                } else
                    for (int ldofy = 1; ldofy <= internalDofCountPerEdge; ++ldofy) {
                        ldof++;
                        if ( ldofy % 2 == 1 )
                            acc( globalDofWeights, ldof )=1.0;
                        else
                            acc( globalDofWeights, ldof )=orientation;

                        acc(globalDofs, ldof) = -1;
                        ++acc(ldofAccessCounts, ldof);
                    }

                edgeIndex = indexSet.subEntityIndex(element, 0, edgeCodim);

                edge=0;
                orientation= ( vnums[edgeToVertexMap[edge][0]] > vnums[edgeToVertexMap[edge][1]]  ) ? -1:1;
                orientation*=m_triangleOrientation[edge];

                dofPosition = 0.5 * (vertices.col(0) + vertices.col(1));
                if (acc(edgeStartingGlobalDofs, edgeIndex) >= 0 &&
                        elementContained) {
                    start = acc(edgeStartingGlobalDofs, edgeIndex);
                    end = start + internalDofCountPerEdge;
                    step = 1;
                    for (int gdof = start; gdof != end; gdof += step) {
                        ldof++;

                        if ( gdof-start % 2 == 0 )
                            acc( globalDofWeights, ldof )=1.0;
                        else
                            acc( globalDofWeights, ldof )=orientation;;

                        acc(globalDofs, ldof) = gdof;
                        acc(m_global2localDofs, gdof).push_back(LocalDof(elementIndex, ldof));
                        ++acc(ldofAccessCounts, ldof);
                        ++acc(gdofAccessCounts, gdof);
                        extendBoundingBox(acc(m_globalDofBoundingBoxes, gdof), vertices);
                        setBoundingBoxReference<CoordinateType>(
                            acc(m_globalDofBoundingBoxes, gdof), dofPosition);
                        ++m_flatLocalDofCount;
                    }
                } else
                    for (int ldofy = 1; ldofy <= internalDofCountPerEdge; ++ldofy) {
                        ldof++;
                        if ( ldofy % 2 == 1 )
                            acc( globalDofWeights, ldof )=1.0;
                        else
                            acc( globalDofWeights, ldof )=orientation;

                        acc(globalDofs, ldof) = -1;
                        ++acc(ldofAccessCounts, ldof);
                    }
            }

            // bubble dofs
            if (m_polynomialOrder >= 3) {
                dofPosition = (vertices.col(0) + vertices.col(1) +
                               vertices.col(2)) / 3.;
                bool useDofs = acc(bubbleStartingGlobalDofs, elementIndex) >= 0;
                for (int ldofy = 1, gdof = acc(bubbleStartingGlobalDofs, elementIndex);
                     ldofy < m_polynomialOrder; ++ldofy)
                    for (int ldofx = 1; ldofx + ldofy < m_polynomialOrder;
                         ++ldofx, ++gdof) {
                        ldof++;
                        acc( globalDofWeights, ldof )=1.0;

                        if (useDofs) {
                            acc(globalDofs, ldof) = gdof;
                            acc(m_global2localDofs, gdof).push_back(
                                LocalDof(elementIndex, ldof));
                            ++acc(gdofAccessCounts, gdof);
                            extendBoundingBox(acc(m_globalDofBoundingBoxes, gdof), vertices);
                            setBoundingBoxReference<CoordinateType>(
                                acc(m_globalDofBoundingBoxes, gdof), dofPosition);
                            ++m_flatLocalDofCount;
                        }
                        else
                            acc(globalDofs, ldof) = -1;
                        ++acc(ldofAccessCounts, ldof);
                    }
            }
            for (size_t i = 0; i < ldofAccessCounts.size(); ++i)
            {
                assert(acc(ldofAccessCounts, i) == 1);
            }
        } else
            throw std::runtime_error("NgTraceSpace::"
                                     "assignDofsImpl(): quadrilateral elements "
                                     "are not supported yet");

        it->next();
    }

    m_vertexGlobalDofs=vertexGlobalDofs;
    m_edgeStartingGlobalDofs=edgeStartingGlobalDofs;
    m_bubbleStartingGlobalDofs=bubbleStartingGlobalDofs;
    // for (size_t i = 0; i < gdofAccessCounts.size(); ++i)
    //     std::cout << i << " " << acc(gdofAccessCounts, i) << "\n";

#ifndef NDEBUG
    for (size_t i = 0; i < globalDofCount_; ++i) {
        const BoundingBox<CoordinateType>& bbox = acc(m_globalDofBoundingBoxes, i);

        assert(bbox.reference.x >= bbox.lbound.x);
        assert(bbox.reference.y >= bbox.lbound.y);
        assert(bbox.reference.z >= bbox.lbound.z);
        assert(bbox.reference.x <= bbox.ubound.x);
        assert(bbox.reference.y <= bbox.ubound.y);
        assert(bbox.reference.z <= bbox.ubound.z);
    }
#endif // NDEBUG

    // Initialize the container mapping the flat local dof indices to
    // local dof indices
    SpaceHelper<BasisFunctionType>::initializeLocal2FlatLocalDofMap(
                m_flatLocalDofCount, m_local2globalDofs, m_flatLocal2localDofs);


    if ( m_polynomialOrder >3 ) //we dont need weights for higher order
    {
        std::cout<<"dropping weights, as they are not needed for order>3"<<std::endl;
        m_local2globalDofWeights.clear();
    }

}

template <typename BasisFunctionType>
size_t NgTraceSpace<BasisFunctionType>::globalDofCount() const
{
    return m_global2localDofs.size();
}

template <typename BasisFunctionType>
size_t NgTraceSpace<BasisFunctionType>::flatLocalDofCount() const
{
    return m_flatLocalDofCount;
}

template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::getGlobalDofs(
    const Entity<0>& element, std::vector<GlobalDofIndex>& dofs, std::vector< BasisFunctionType > &  localDofWeights ) const
{
    const Mapper& mapper = m_view->elementMapper();
    EntityIndex index = mapper.entityIndex(element);
    dofs = m_local2globalDofs[index];
    localDofWeights.resize(dofs.size());

    if ( m_polynomialOrder <= 3 )
    {
        localDofWeights=m_local2globalDofWeights[index];
    }else //since we use a different shapeset for each kind of element, we can just use 1 as weight
    {
        for ( int i=0;i<dofs.size();i++ )
            localDofWeights[i]=1;

    }

}

template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::getGlobalInnerDofs(
        const Entity<0>& element, std::vector<GlobalDofIndex>& dofs) const
{
    const IndexSet& iset = m_view->indexSet();
    const int internalDofCountPerEdge = m_polynomialOrder - 1;

    EntityIndex index = iset.entityIndex(element);
    const int bubbleDofCountPerTriangle =
        std::max(0, (m_polynomialOrder - 1) * (m_polynomialOrder - 2) / 2);


    const int start=m_bubbleStartingGlobalDofs[index];
    for(int i=0;i<bubbleDofCountPerTriangle;i++)
        dofs.push_back(start+i);

  /*
  //TODO this is actually wrong
  const std::vector<GlobalDofIndex>& v=m_local2globalDofs[index];
  dofs.insert(dofs.end(), v.begin(), v.end());*/
}


template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::getGlobalEdgeDofs(
										    const Entity<1>& element, std::vector<GlobalDofIndex>& dofs) const
{
  const IndexSet& iset = m_view->indexSet();
  const int internalDofCountPerEdge = m_polynomialOrder - 1;

  EntityIndex index = iset.entityIndex(element);

  const int start=m_edgeStartingGlobalDofs[index];
  for(int i=0;i<internalDofCountPerEdge;i++)
    dofs.push_back(start+i);
}

template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::getGlobalVertexDofs(
										      const Entity<2>& element, std::vector<GlobalDofIndex>& dofs) const
{
  const IndexSet& iset = m_view->indexSet();
  EntityIndex index = iset.entityIndex(element);

  dofs.push_back(m_vertexGlobalDofs[index]);
}


template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::global2localDofs(
        const std::vector<GlobalDofIndex>& globalDofs,
        std::vector<std::vector<LocalDof> >& localDofs,
        std::vector<std::vector<BasisFunctionType> >& localDofWeights
    ) const
{
    localDofs.resize(globalDofs.size());
    localDofWeights.resize( globalDofs.size() );
    if ( m_polynomialOrder<=3 )
    {
        for (size_t i = 0; i < globalDofs.size(); ++i)
        {
            localDofs[i] = m_global2localDofs[globalDofs[i]];
            std::vector<BasisFunctionType>& activeLdofWeights = acc( localDofWeights,  i );
            activeLdofWeights.resize( localDofs[i].size() );
            for ( size_t j = 0; j < localDofs[i].size(); ++j ) {
                LocalDof ldof = acc( localDofs[i],  j );
                acc( activeLdofWeights,  j ) = acc( acc( m_local2globalDofWeights,
                                                         ldof.entityIndex ),
                                                    ldof.dofIndex );
            }

        }
    }else
    {
        for (size_t i = 0; i < globalDofs.size(); ++i)
        {
            localDofs[i] = m_global2localDofs[globalDofs[i]];
            std::vector<BasisFunctionType>& activeLdofWeights = acc( localDofWeights,  i );
            activeLdofWeights.resize( localDofs[i].size() );
            for ( size_t j = 0; j < localDofs[i].size(); ++j ) {
                activeLdofWeights[j]=1.0;
            }
        }
    }
}

template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::flatLocal2localDofs(
        const std::vector<FlatLocalDofIndex>& flatLocalDofs,
        std::vector<LocalDof>& localDofs) const
{
    localDofs.resize(flatLocalDofs.size());
    for (size_t i = 0; i < flatLocalDofs.size(); ++i)
        localDofs[i] = m_flatLocal2localDofs[flatLocalDofs[i]];
}

template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::getGlobalDofPositions(
        std::vector<Point3D<CoordinateType> >& positions) const
{
    positions.resize(m_globalDofBoundingBoxes.size());
    for (size_t i = 0; i < m_globalDofBoundingBoxes.size(); ++i)
        acc(positions, i) = acc(m_globalDofBoundingBoxes, i).reference;
}

template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::getFlatLocalDofPositions(
        std::vector<Point3D<CoordinateType> >& positions) const
{
    throw std::runtime_error("NgTraceSpace::"
                             "getFlatLocalDofPositions(): not implemented yet");
}

template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::getGlobalDofBoundingBoxes(
    std::vector<Bempp::BoundingBox<CoordinateType> >& bboxes) const
{
    bboxes = m_globalDofBoundingBoxes;
}

template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::
getFlatLocalDofBoundingBoxes(
    std::vector<Bempp::BoundingBox<CoordinateType> >& bboxes) const
{
    throw std::runtime_error("NgTraceSpace::"
                             "getFlatLocalDofBoundingBoxes(): not implemented yet");
}

template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::getGlobalDofNormals(
        std::vector<Point3D<CoordinateType> >& normals) const
{
    SpaceHelper<BasisFunctionType>::
            getGlobalDofNormals_defaultImplementation(
                *m_view, m_global2localDofs, normals);
}

template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::getFlatLocalDofNormals(
        std::vector<Point3D<CoordinateType> >& normals) const
{
    throw std::runtime_error("NgTraceSpace::"
                             "getFlatLocalDofNormals(): not implemented yet");
}

template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::dumpClusterIds(
        const char* fileName,
        const std::vector<unsigned int>& clusterIdsOfDofs) const
{
    dumpClusterIdsEx(fileName, clusterIdsOfDofs, GLOBAL_DOFS);
}

template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::dumpClusterIdsEx(
        const char* fileName,
        const std::vector<unsigned int>& clusterIdsOfDofs,
        DofType dofType) const
{
    throw std::runtime_error("NgTraceSpace::"
                             "dumpClusterIdsEx(): not implemented yet");
}


template <typename BasisFunctionType>
const ngstd::Array<int>& NgTraceSpace<BasisFunctionType>::bemppToNgMap() const
{
    return m_bemppToNg;
}

template <typename BasisFunctionType>
const ngstd::Array<int>& NgTraceSpace<BasisFunctionType>::ngToBemppMap() const
{
    return m_ngToBempp;
}

template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::updateNgMaps()
{
    std::cout<<"updating ng maps...."<<std::endl;

    auto ma=m_fespace->GetMeshAccess();
    int nse=ma->GetNSE();
    Array<int> dnums;

    std::vector<Bempp::GlobalDofIndex> bemDofs;
    std::vector<BasisFunctionType> bemWeights;

    std::cout<<"nse= "<<nse<<" vs "<<m_view->entityCount( 0 )<<std::endl;

    const int dimNg=m_fespace->GetNDof();
    const int dimBem=globalDofCount();

    m_ngToBempp.SetSize(dimNg);
    m_ngToBempp=-1; //not all ng Dofs have a bempp counterpart. -1 if no match exists
    m_bemppToNg.SetSize( dimBem );
    m_bemppToNg=-1;

    std::cout<<"dimNg: "<<dimNg<<" dimBem "<<dimBem<<std::endl;

    if(nse!=m_view->entityCount(0))
    {
        throw Exception ( "The Bempp mesh doesn't match the NGSolve one");
    }
    const Bempp::ReverseElementMapper& invMap=m_view->reverseElementMapper();
    double maxE=0;
    Array<int> vnums;
    Vec<3> p;
    Bempp::Matrix<double> corners;
    Bempp::Matrix<double> g2l;
    for (int i = 0; i < nse; i++)
    {
        m_fespace->GetSDofNrs (i, dnums);
        //only used for some debugging

        ma->GetSElVertices ( i,   vnums );

        const Bempp::EntityPointer<0>& entp=invMap.entityPointer(i);
        const Bempp::Entity<0>& ent=entp.entity();


        ent.geometry().getCorners( corners );

        ent.geometry().global2local( corners, g2l );
        /*std::cout<<"--"<<std::endl;
        std::cout<<g2l;
        std::cout<<"--"<<std::endl;*/

        for ( int j= 0;j<corners.cols();j++ )
        {
            //std::cout<<"v: "<<vnums[j]<<" to el "<<dnums[j]<<std::endl;
            ma->GetPoint( vnums[ j ],p );

            for ( int l=0;l<3;l++ )
            {
                const double e=fabs(  p[l] - corners( l, j ) );
                maxE=std::max( e, maxE );
                if ( e > 1E-10 )
                {
                    std::cout<<"error at j="<<j<<" l="<<l<<" e="<<fabs(  p[l] - corners( l, j ) )<<std::endl;
                    std::cout<<p<<"\n"<<corners.col( j )<<std::endl;
                    throw Exception( "The meshes do not match!" );
                }
            }
        }

        bemDofs.clear();
        bemWeights.clear();
        getGlobalDofs(ent,bemDofs, bemWeights);

        //std::cout<<"bem: "<<bemDofsDual.size()<<" vs "<<dnums.Size()<<std::endl;

        if ( bemDofs.size()!=dnums.Size() )
            throw Exception( "The elements range DOFs don't match" );

        for ( int j=0;j<dnums.Size();j++ )
        {
            if ( m_bemppToNg[bemDofs[j]] < 0 )
                m_bemppToNg[bemDofs[j]]=dnums[j];
            else
            {
                if ( m_bemppToNg[bemDofs[j]] != dnums[j] )
                    std::cout<<"overwriting: "<<m_bemppToNg[bemDofs[j]]<<" with "<<dnums[j]<<std::endl;
            }
            m_ngToBempp[dnums[j]]=bemDofs[j];
        }


            /*for ( int j=0;j<bemDofs.size();j++ )
        {
            m_bemppToNg[bemDofs[j]]=dnums[j];
            //std::cout<<m_bemppToNg[bemDofs[j]]<<" vs "<<dnums[j]<<std::endl;
            m_ngToBempp[dnums[j]]=bemDofs[j];

            }*/
    }

    //check if we managed to match every bem dof
    for ( int i=0;i<m_bemppToNg.Size();i++ )
    {
        assert( m_bemppToNg[i] !=-1 );
    }


    std::cout<<"geometry error: "<<maxE<<std::endl;

}


template <typename BasisFunctionType>
void NgTraceSpace<BasisFunctionType>::updateShapesets()
{
    std::cout<<"rebuilding shapesets..:"<<std::endl;
    const Bempp::IndexSet& iset = m_view->indexSet();
    int cnt=0;
    std::unique_ptr<EntityIterator<0> > it = m_view->entityIterator<0>();
    while (!it->finished())
    {
        const Bempp::Entity<0>& entity=it->entity();
        Bempp::EntityIndex element_id = iset.entityIndex(entity);
        int orientation=computeOrientationFlag( entity );
        Fiber::Shapeset<BasisFunctionType>* found_shapeset=findShapeset( orientation );

        if ( found_shapeset==0 )
        {
            std::cout<<"new shapeset at "<<orientation<<std::endl;
            Fiber::Shapeset<BasisFunctionType>* shapeset=( new ( *m_spaceHeap )  NgTraceShapeset<BasisFunctionType>(element_id, m_fespace, * m_spaceHeap) );
            m_shapesets.push_back( std::make_pair( orientation, shapeset ) );
            cnt++;
        }

        it->next();
    }
    std::cout<<"i built "<<cnt<<" flavors of elements"<<std::endl;

}




template class NgTraceSpace<double >;
template class NgTraceSpace<Complex >;
//FIBER_INSTANTIATE_CLASS_TEMPLATED_ON_BASIS(NgTraceSpace);
