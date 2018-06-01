#ifndef BLOCK_MATRIX_H
#define BLOCK_MATRIX_H





#include "assembly/discrete_boundary_operator.hpp"
#include <Eigen/Core>


#include "solve.hpp"

#include <tbb/parallel_for.h>
#include <tbb/spin_mutex.h>


#include "ng_trace_space.hpp"


namespace NgBem{

using ngla::BaseVector;
using ngla::BaseMatrix;
using ngbla::Complex;
using ngbla::BitArray;
using ngbla::Exception;
using ngla::VVector;
using ngla::AutoVector;

template<typename T>
class BemppBlock : public BaseMatrix
{
public:
    BemppBlock(boost::shared_ptr<const Bempp::DiscreteBoundaryOperator<T> > op, bool domIsNg,
               bool dualIsNg, const boost::shared_ptr<const NgTraceSpace<double> > ngspace, unsigned int blocks=1);

    ~BemppBlock();

    /// virtual function must be overloaded
    virtual int VHeight() const;

    /// virtual function must be overloaded
    virtual int VWidth() const;

    /// linear access of matrix memory
    virtual BaseVector& AsVector()
    {
        throw Exception( "Not implemented" );
    }
    /// linear access of matrix memory
    virtual const BaseVector & AsVector() const
    {
        throw Exception( "Not implemented" );
    }

    boost::shared_ptr<const Bempp::DiscreteBoundaryOperator<T> > bemOperator() const
    {
        return m_op;
    }

    virtual bool IsComplex() const;


    //virtual ostream & Print (ostream & ost) const;
    //virtual void MemoryUsage (ngstd::Array<MemoryUsageStruct*> & mu) const;

    // virtual const void * Data() const;
    // virtual void * Data();

    /// creates matrix of same type
    //virtual BaseMatrix * CreateMatrix () const {};
    /// creates matrix of same type
    // virtual BaseMatrix * CreateMatrix (const ngstd::Array<int> & elsperrow) const;
    /// creates a matching vector, size = width
    //virtual AutoVector * CreateRowVector () const;
    /// creates a matching vector, size = height
    //virtual AutoVector * CreateColVector () const;
    /// creates a matching vector (for square matrices)
    //virtual AutoVector * CreateVector () const;

    /// y = matrix * x. Multadd should be implemented, instead
    //virtual void Mult (const BaseVector & x, BaseVector & y) const;
    /// y += s matrix * x
    virtual void MultAdd (double s, const BaseVector & x, BaseVector & y) const;
    /// y += s matrix * x
    virtual void MultAdd (Complex s, const BaseVector & x, BaseVector & y) const;

    /// y += s Trans(matrix) * x
    virtual void MultTransAdd (double s, const BaseVector & x, BaseVector & y) const;
    /// y += s Trans(matrix) * x
    virtual void MultTransAdd (Complex s, const BaseVector & x, BaseVector & y) const;

private:
    template <typename S>
    void MultAddImpl( S s, const BaseVector& x, BaseVector& y ) const;


    /**
       to split mat x vec for symmetric matrices
       only rows with inner or cluster true need by added (but more can be ...)
    */
    /*virtual void MultAdd1 (double s, const BaseVector & x, BaseVector & y,
			   const BitArray * ainner = NULL,
			   const ngstd::Array<int> * acluster = NULL) const;

    /// only cols with inner or cluster true need by added (but more can be ...)
    virtual void MultAdd2 (double s, const BaseVector & x, BaseVector & y,
			   const BitArray * ainner = NULL,
			   const ngstd::Array<int> * acluster = NULL) const;

    */
    //void SetParallelDofs (const ParallelDofs * pardofs) { paralleldofs = pardofs; }
    //const ParallelDofs * GetParallelDofs () const { return paralleldofs; }

    /*virtual BaseMatrix * InverseMatrix (const Bitngstd::Array * subset = 0) const;
    virtual BaseMatrix * InverseMatrix (const ngstd::Array<int> * clusters) const;
    virtual INVERSETYPE SetInverseType ( INVERSETYPE ainversetype ) const;
    virtual INVERSETYPE SetInverseType ( string ainversetype ) const;
    virtual INVERSETYPE  GetInverseType () const;*/

    shared_ptr<BaseMatrix> InverseMatrix ( const BitArray * subset = 0 ) const;

private:
    boost::shared_ptr<const NgTraceSpace<double> > m_ngSpace;
    boost::shared_ptr<const Bempp::DiscreteBoundaryOperator<T> > m_op;
    bool m_domIsNg;
    bool m_dualIsNg;
    unsigned int m_blocks;
};

class NullBlock : public BaseMatrix
{
public:
    NullBlock(unsigned int w, unsigned int h )
    {
        m_width=w;
        m_height=h;
    }
    ~NullBlock()
    {

    }

    virtual bool IsComplex() const
    {
        return false;
    }


    /// virtual function must be overloaded
    virtual int VHeight() const
    {
        return m_height;
    }

    /// virtual function must be overloaded
    virtual int VWidth() const
    {
        return m_width;
    }

    /// linear access of matrix memory
    virtual BaseVector & AsVector()
    {
        throw Exception( "Not implemented" );
    }
    /// linear access of matrix memory
    virtual const BaseVector & AsVector() const
    {
        throw Exception( "Not implemented" );
    }

    virtual void MultAdd (double s, const BaseVector & x, BaseVector & y) const
    {
        //Do nothing
    }
    /// y += s matrix * x
    virtual void MultAdd (Complex s, const BaseVector & x, BaseVector & y) const
    {
        //Do nothing
    }

    /// y += s Trans(matrix) * x
    virtual void MultTransAdd (double s, const BaseVector & x, BaseVector & y) const
    {
        //Do Nothing
    }
    /// y += s Trans(matrix) * x
    virtual void MultTransAdd (Complex s, const BaseVector & x, BaseVector & y) const
    {
        //Do nothing
    }

    virtual AutoVector CreateVector () const
    {
        throw Exception( "Wtf?" );
    }

private:
    unsigned int m_width;
    unsigned int m_height;
};



template <typename T>
class BlockMatrix : public BaseMatrix
{

public:
    BlockMatrix(unsigned  int rows, unsigned int cols);
    ~BlockMatrix();

    void setBlock( unsigned int i, unsigned int j, shared_ptr<const BaseMatrix> mat, bool ownBlock=false );
    shared_ptr<const BaseMatrix> block( unsigned int i, unsigned int j ) const;

    virtual void MultAdd ( double s,  const BaseVector & x,  BaseVector & y ) const;
    virtual void MultAdd ( Complex s,  const BaseVector & x,  BaseVector & y ) const;

    /// y += s Trans(matrix) * x
    virtual void MultTransAdd (double s, const BaseVector & x, BaseVector & y) const;
    /// y += s Trans(matrix) * x
    virtual void MultTransAdd (Complex s, const BaseVector & x, BaseVector & y) const;


    virtual shared_ptr<BaseMatrix> InverseMatrix(const BitArray* subset=0 ) const;

    virtual AutoVector CreateVector () const;
    virtual AutoVector CreateRowVector () const
    {
        return CreateVector();
    }

    virtual AutoVector CreateColVector () const
    {
        return CreateVector();
    }

        /// virtual function must be overloaded
    virtual int VHeight() const;

    /// virtual function must be overloaded
    virtual int VWidth() const;

    virtual bool IsComplex() const
    {
        return Eigen::NumTraits<T>::IsComplex;
    }




    //IntRange GetRange( int block ) const;

protected:
    unsigned int m_rows, m_cols;
    Array<shared_ptr<const BaseMatrix> > m_blocks;
};


class WrapperBlock : public BaseMatrix
{
public:
    WrapperBlock(int w, int h,shared_ptr<const BaseMatrix> mat, shared_ptr<BaseMatrix> precond=shared_ptr<BaseMatrix>() )
    {
        m_width=w;
        m_height=h;
        m_mat=mat;
        m_precond=precond;
    }
    ~WrapperBlock()
    {
        //delete m_mat;
    }

    /// virtual function must be overloaded
    virtual int VHeight() const
    {
        return m_height;
    }

    /// virtual function must be overloaded
    virtual int VWidth() const
    {
        return m_width;
    }

/*    /// linear access of matrix memory
    virtual BaseVector & AsVector()
    {
        return ;//m_mat->AsVector();
        }*/
    /// linear access of matrix memory
    virtual const BaseVector & AsVector() const
    {
        return m_mat->AsVector();
    }

    virtual AutoVector CreateVector () const
    {
        return m_mat->CreateVector();
    }

    virtual AutoVector CreateRowVector () const
    {
        return CreateVector();
    }

    virtual AutoVector CreateColVector () const
    {
        return CreateVector();
    }

    virtual void MultAdd (double s, const BaseVector & x, BaseVector & y) const
    {
        m_mat->MultAdd( s, x, y );
        //Do nothing
    }
    /// y += s matrix * x
    virtual void MultAdd (Complex s, const BaseVector & x, BaseVector & y) const
    {
        m_mat->MultAdd( s, x, y );
        //Do nothing
    }

    /// y += s Trans(matrix) * x
    virtual void MultTransAdd (double s, const BaseVector & x, BaseVector & y) const
    {
        m_mat->MultTransAdd( s, x, y );
        //Do Nothing
    }
    /// y += s Trans(matrix) * x
    virtual void MultTransAdd (Complex s, const BaseVector & x, BaseVector & y) const
    {
        m_mat->MultTransAdd( s, x, y );
        //Do nothing
    }

    virtual shared_ptr<BaseMatrix> InverseMatrix(const BitArray* subset=0 ) const
    {
        if ( m_precond )
        {
            return m_precond;
        }else
        {
	//   return m_mat->InverseMatrix( subset );
        }
    }

private:
    int m_width;
    int m_height;
    shared_ptr<const BaseMatrix>  m_mat;
    shared_ptr<BaseMatrix> m_precond;
};

template <typename T>
class SumMatrixExpr : public BaseMatrix
{
public:
    SumMatrixExpr(const ngstd::Array<std::shared_ptr<BaseMatrix> >& mat )
    {
        m_mat=mat;
    }
    ~SumMatrixExpr()
    {
        std::cout<<"sum expr is dying"<<std::endl;
        //leak some memory!!
        /*for ( int i=0;i<m_size;i++ )
        {
            delete m_mat[i];
            m_mat[i]=0;
            }*/
    }

    /// virtual function must be overloaded
    virtual int VHeight() const
    {
        return m_mat[0]->Height();
    }

    /// virtual function must be overloaded
    virtual int VWidth() const
    {
        return m_mat[0]->Width();
    }

/*    /// linear access of matrix memory
    virtual BaseVector & AsVector()
    {
        return ;//m_mat->AsVector();
        }*/
    /// linear access of matrix memory
    virtual const BaseVector & AsVector() const
    {
        throw Exception( "not implemented" );
        //return m_mat->AsVector();
    }

    virtual AutoVector CreateVector () const
    {
        return m_mat[0]->CreateVector();
    }

    virtual AutoVector CreateRowVector () const
    {
        return CreateVector();
    }

    virtual AutoVector CreateColVector () const
    {
        return CreateVector();
    }


    virtual void MultAdd (double s, const BaseVector & x, BaseVector & y) const
    {
        MultAddImpl( s, x, y );
        /*for ( int i=0;i<m_size;i++ )
        {
            m_mat[i]->MultAdd( s, x, y );
            }*/
    }
    /// y += s matrix * x
    virtual void MultAdd (Complex s, const BaseVector & x, BaseVector & y) const
    {
        MultAddImpl( s, x, y );
        /*for ( int i=0;i<m_size;i++ )
          m_mat[i]->MultAdd( s, x, y );*/
    }

    template<typename TS>
    inline void MultAddImpl( TS s, const BaseVector& x,  BaseVector& y ) const
    {

        typedef tbb::spin_mutex MultAddLoopMutexType;
        MultAddLoopMutexType mutex;
        tbb::parallel_for ( tbb::blocked_range<size_t>( 0,m_mat.Size() ), [this, &x, &y, &mutex, s]( tbb::blocked_range<size_t> r )
        {
            auto tmp=y.CreateVector();
            *tmp=0;
            for ( size_t l=r.begin();l!=r.end();l++ )
            {
                this->m_mat[l]->MultAdd( s, x, *tmp );
            }

            //Update the result
            {
                MultAddLoopMutexType::scoped_lock lock( const_cast<MultAddLoopMutexType&>(  mutex ) );
                y+=*tmp;
            }
        } );
    }

    /// y += s Trans(matrix) * x
    virtual void MultTransAdd (double s, const BaseVector & x, BaseVector & y) const
    {
        for ( int i=0;i<m_mat.Size();i++ )
            m_mat[i]->MultTransAdd( s, x, y );
    }
    /// y += s Trans(matrix) * x
    virtual void MultTransAdd (Complex s, const BaseVector & x, BaseVector & y) const
    {
        for ( int i=0;i<m_mat.Size();i++ )
            m_mat[i]->MultTransAdd( s, x, y );
    }

    virtual shared_ptr<BaseMatrix> InverseMatrix(const BitArray* subset=0 ) const
    {
        //std::cerr<<"called"<<std::endl;
        //B
        //SparseMatrixTM<Complex> *tmp=dynamic_cast<SparseMatrixTM<Complex> *>( m_mat[1]->CreateMatrix() );
        //tmp->SetInverseType( SPARSECHOLESKY );
        //tmp->AsVector()=m_mat[1].AsVector();

	//  return m_mat[1]->InverseMatrix(subset);
    }

private:
    Array<std::shared_ptr<const BaseMatrix> > m_mat;
};


template <typename TB>
template< typename T>
void BemppBlock<TB>::MultAddImpl( T s, const BaseVector& x, BaseVector& y ) const
{
    Bempp::Vector<T> domTmp(m_op->columnCount());
    Bempp::Vector<T> dualTmp(m_op->rowCount());
    dualTmp.fill( 0 );
    const ngstd::Array<int>& bemppToNg=m_ngSpace->bemppToNgMap();

    if ( m_domIsNg ) //transform x to a scaled down tmp vector
    {
        //std::cout<<"converting domain to"<<m_op->columnCount()<<" from"<<x.Size()<<std::endl;
        size_t blockSizeBem=m_ngSpace->globalDofCount();
        size_t blockSizeNg=m_ngSpace->FESpaceDofCount();
        const ngbla::FlatVector<T>& fv=x.FV<T>();
        for ( unsigned int j=0;j<m_blocks;j++ )
        {
            for ( unsigned int i=0;i<blockSizeBem ; i++ )
            {
                domTmp( i + j*blockSizeBem )  = fv[j*blockSizeNg+bemppToNg[ i ] ];
            }
        }

    }else
    {
        const ngbla::FlatVector<T>& fv=x.FV<T>();
        for ( unsigned int i=0;i<m_op->columnCount() ; i++ )
        {
            domTmp( i )  = fv[ i ];
        }
    }

    m_op->apply( Bempp::NO_TRANSPOSE, domTmp, dualTmp, s, 0 );


    if ( m_dualIsNg ) //transform x to a scaled down tmp vector
    {
        //std::cout<<"converting domain to"<<m_op->columnCount()<<" from"<<x.Size()<<std::endl;
        size_t blockSizeBem=m_ngSpace->globalDofCount();
        size_t blockSizeNg=m_ngSpace->FESpaceDofCount();
        const ngbla::FlatVector<T>& fv=y.FV<T>();
        for ( unsigned int j=0;j<m_blocks;j++ )
        {
            for ( unsigned int i=0;i<blockSizeBem ; i++ )
            {
                fv[j*blockSizeNg+bemppToNg[ i ] ] += dualTmp( i+j*blockSizeBem );
            }
        }

    }else
    {
        const ngbla::FlatVector<T>& fv=y.FV<T>();
        for ( unsigned int i=0;i<m_op->rowCount() ; i++ )
        {
            fv( i )+=dualTmp( i );
        }
    }

}


}

#endif
