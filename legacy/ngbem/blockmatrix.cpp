#include "blockmatrix.hpp"

#include <tbb/parallel_for.h>

namespace NgBem{

template<typename T>
BemppBlock<T>::BemppBlock(boost::shared_ptr<const Bempp::DiscreteBoundaryOperator<T> > op, bool domIsNg,
                          bool dualIsNg,  boost::shared_ptr<const NgTraceSpace<double> > ngspace, unsigned int blocks ) :
    m_op( op ),
    m_domIsNg( domIsNg ),
    m_dualIsNg( dualIsNg ),
    m_ngSpace( ngspace ),
    m_blocks( blocks )
{

}

template<typename T>
BemppBlock<T>::~BemppBlock()
{
    std::cout<<"bemblock dying"<<std::endl;
}

template<typename T>
bool BemppBlock<T>::IsComplex() const
{
    return Eigen::NumTraits<T>::IsComplex;
}


/// virtual function must be overloaded
template<typename T>
int BemppBlock<T>::VHeight() const
{
    if ( m_dualIsNg )
        return m_blocks*m_ngSpace->FESpaceDofCount();
    else
        return m_op->rowCount();
}

/// virtual function must be overloaded
template<typename T>
int BemppBlock<T>::VWidth() const
{
    if ( m_domIsNg )
        return m_blocks*m_ngSpace->FESpaceDofCount();
    else
        return m_op->columnCount();
}

/// y += s matrix * x
template<>
void BemppBlock<double>::MultAdd (double s, const BaseVector & x, BaseVector & y) const
{
    MultAddImpl<double>( s,x,y );
}

/// y += s matrix * x
template<typename T>
void BemppBlock<T>::MultAdd (Complex s, const BaseVector & x, BaseVector & y) const
{
    throw Exception( "Complex not implemented!" );
}

template<>
void BemppBlock<Complex>::MultAdd (Complex s, const BaseVector & x, BaseVector & y) const
{
    MultAddImpl<Complex>( s, x, y );
}

template<>
void BemppBlock<Complex>::MultAdd (double s, const BaseVector & x, BaseVector & y) const
{
    MultAdd( Complex( s, 0.0 ), x, y );
}



/// y += s Trans(matrix) * x
template<>
void BemppBlock<double>::MultTransAdd (double s, const BaseVector & x, BaseVector & y) const
{
    throw Exception( "Transpose not implemented" );
}

template<>
void BemppBlock<Complex>::MultTransAdd (Complex s, const BaseVector & x, BaseVector & y) const
{
    throw Exception( "MultTransAdd not implemented" );
}

/// y += s Trans(matrix) * x
template<typename T>
void BemppBlock<T>::MultTransAdd (Complex s, const BaseVector & x, BaseVector & y) const
{
    throw Exception( "Complex MultTransAdd not supported" );
}

template<>
void BemppBlock<Complex>::MultTransAdd (double s, const BaseVector & x, BaseVector & y) const
{
    throw Exception( "Trans!" );


}

template <typename T>
shared_ptr<BaseMatrix> BemppBlock<T>::InverseMatrix( const BitArray* subset) const
{
  throw Exception("Not implemented");
}

template<>
shared_ptr<BaseMatrix> BemppBlock<Complex>::InverseMatrix ( const BitArray * subset) const
{
    throw Exception( "Not implemented" );
}






template <typename T>
BlockMatrix<T>::BlockMatrix( unsigned int rows, unsigned int cols) : m_rows( rows ), m_cols( cols ),  m_blocks( rows*cols )
{
    std::cout<<"building a "<<rows<<"x"<<cols<<" block matrix"<<std::endl;

}

template <typename T>
BlockMatrix<T>::~BlockMatrix()
{
    std::cout<<"blockm dying"<<std::endl;
}

template <typename T>
int BlockMatrix<T>::VWidth() const
{
    unsigned int w=0;
    for ( unsigned int i=0;i<m_cols;i++ )
        w+=m_blocks[i]->Width();

    return w;
}


template <typename T>
int BlockMatrix<T>::VHeight() const
{
    unsigned int h=0;
    for ( unsigned int i=0;i<m_rows;i++ )
        h+=m_blocks[i*m_cols]->Height();

    return h;
}

template <typename T>
void BlockMatrix<T>::MultAdd ( double s,  const BaseVector & x,  BaseVector & y ) const
{
  //unsigned int rowStartIdx=0;
  //unsigned int colStartIdx=0;
  //const BaseMatrix* block=0;

  AutoVector tmpy[m_cols];
  for(int i=0;i<m_cols;i++)
  {
    tmpy[i].AssignPointer( y.CreateVector() );
    (*tmpy[i])=0;
  }
  //tbb::parallel_for (
  //    tbb::blocked_range<size_t>( 0, m_rows ),
  //    [&x,&y,&tmpy,s,this]( const tbb::blocked_range<size_t>& r)
  tbb::blocked_range<size_t> r( 0, m_rows );
      {
	for ( size_t i=r.begin(); i!=r.end(); ++i )
	  {
	    unsigned int rowStartIdx=0;
	    for(int j=0;j<i;j++)
	      rowStartIdx+=m_blocks[j*m_cols]->Height();

	    //tbb::parallel_for(tbb::blocked_range<size_t>( 0, m_cols ),
            //		      [&x,&tmpy,s,this,i,rowStartIdx](const tbb::blocked_range<size_t>& r2)
            tbb::blocked_range<size_t> r2( 0, m_cols );
	    {
	      for ( unsigned int j=r2.begin();j<r2.end();j++ )
		{
		  int colStartIdx=0;
		  for(int l=0;l<j;l++)
		    colStartIdx+=m_blocks[i*m_cols+l]->Width();

		  shared_ptr<const BaseMatrix>& block=m_blocks[i*m_cols+j];

		  if ( block==0 )
                      continue;

		  const unsigned int bw=block->Width();
		  const unsigned int bh=block->Height();

		  auto rx=x.Range(colStartIdx, colStartIdx+bw);
		  auto ry=tmpy[j].Range(rowStartIdx, rowStartIdx+bh);

		  //BaseVector* tmpx=rx->CreateVector();
		  //T w=m_weights[i*m_cols+j];
		  //*tmpx=*rx*w;

		  block->MultAdd(s, *rx, *ry );

                  //std::cout<<"ry: "<<ry.L2Norm()<<std::endl;

		  //delete tmpx;

		  //delete rx;
		  //delete ry;
		}
	    } //);
	  }
      }//);

  for(int i=0;i<m_cols;i++)
    {
        //std::cout<<"adding "<<( tmpy[i] ).L2Norm()<<std::endl;
        y+=*(tmpy[i]);
        //delete tmpy[i];
    }

}

template <typename T>
void BlockMatrix<T>::MultAdd ( Complex s,  const BaseVector & x,  BaseVector & y ) const
{
    std::cout<<"multadd c"<<std::endl;
    throw Exception( "Complex not impelemented" );
}

/// y += s Trans(matrix) * x
template <typename T>
void BlockMatrix<T>::MultTransAdd (double s, const BaseVector & x, BaseVector & y) const
{
    throw Exception( "TransAdd" );

}

/// y += s Trans(matrix) * x
template <typename T>
void BlockMatrix<T>::MultTransAdd (Complex s, const BaseVector & x, BaseVector & y) const
{
    throw Exception( "Complex not implemented yet" );
}

template <typename T>
AutoVector BlockMatrix<T>::CreateVector () const
{
    std::cout<<"creating a vector"<<std::endl;
    unsigned int h=0;
    for ( unsigned int j=0;j<m_cols ;j++ )
    {
        h+=m_blocks[j]->Width();;
    }
    std::cout<<"h: "<<h<<std::endl;

    auto vec=std::make_shared<VVector<T> >(h);
    *vec=0;
    return vec;
}



template <typename T>
void BlockMatrix<T>::setBlock( unsigned int i, unsigned int j, shared_ptr<const BaseMatrix> mat, bool ownsBlock)
{
    m_blocks[i*m_cols+j]=mat;
    //m_ownsBlock[i*m_cols+j]=ownsBlock;
}


template <typename T>
shared_ptr<const BaseMatrix> BlockMatrix<T>::block( unsigned int i, unsigned int j ) const
{
    return m_blocks[i*m_cols+j];
}

template <typename T>
shared_ptr<BaseMatrix> BlockMatrix<T>::InverseMatrix( const BitArray * subset  ) const
{
    std::cerr<<"WARNING THIS WILL ONLY GIVE YOU A DIAGONAL PRECONDITIONER"<<std::endl;
    //throw Exception( "Not implemented" );
    //
    assert( m_rows==m_cols );
    shared_ptr<BlockMatrix> prec=make_shared<BlockMatrix>(m_rows, m_cols);

    for ( unsigned int i=0; i<m_rows;i++ )
    {
        for ( unsigned int j=0;j<m_cols;j++ )
        {
            unsigned int idx=i*m_cols+j;
            shared_ptr<BaseMatrix> inv_block;
            shared_ptr< const BaseMatrix> block=m_blocks[idx];
            if ( i==j ) //invert the diagonal blocks
            {
                //inv_block=block->InverseMatrix(subset);
            }
            else
            {
                inv_block=make_shared<NullBlock>(block->Width(), block->Height() );
            }

            prec->setBlock( i, j, inv_block, true );
        }
    }

    return prec;
    //BiCGStabSolver<T>* solve=new BiCGStabSolver<T>( *this );
    //GMRESSolver<T>* solve=new GMRESSolver<T>( *this );
    //solve->SetMaxSteps( 1000 );
    //return solve;
}
/*
template <typename T>
IntRange BlockMatrix<T>::GetRange( int block ) const
{
    int start=0;
    for ( int i=0;i<block;i++ )
        start+=m_blocks[i]->Width();
    const int end=start+m_blocks[block]->Width();

    return IntRange( start, end-1 );;
}
*/



template class BlockMatrix<double >;
template class BlockMatrix<ngbla::Complex >;

template class BemppBlock<double >;
template class BemppBlock<ngbla::Complex >;
}
