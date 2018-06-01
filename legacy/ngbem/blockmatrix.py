from scipy.sparse.linalg.interface import LinearOperator as _LinearOperator


class BemppBlock(_LinearOperator):
    def __init__(self,dtype,shape):
        super(_LinearOperator, self).__init__();


    def __init__(self, op):
        super(_LinearOperator, self).__init__()
        weak=op.weak_form();
        self.domIsNg=self._testIfNg(op.domain);
        self.dualIsNg=self._testIfNg(op.dual_to_range);

        shape=weak.shape;
        print(shape)
        if(self.domIsNg):
            dimNg=op.domain.fespace.ndof;
            shape=[shape[0],dimNg]
        if(self.dualIsNg):
            dimNg=op.dual_to_range.fespace.ndof;
            shape=[dimNg,shape[1]]
        print(shape)

        self.shape=shape;
        self.dtype=weak.dtype;

        self.op=op;
        self.blocks=1;

    def _testIfNg(self,space):
        try:
            return space.isNg;
        except AttributeError:
            return False;


    def _copyToBem(self,vec,dest, space):
        if(self._testIfNg(space)):
            ngmap=space.bempp_to_ng_map()
            dest=vec[ngmap]
        else:
            dest=vec;
        return dest;

    def _copyFromBem(self,vec,dest, space):
        if(self._testIfNg(space)):
            ngmap=space.bempp_to_ng_map()
            dest[ngmap]=vec;
        else:
            dest=vec;

        return dest;


    def _matvec(self,vec):
        import numpy as np;
        weak=self.op.weak_form();
        domTmp=np.zeros(weak.shape[1],dtype=weak.dtype);
        dualTmp=np.zeros(weak.shape[0],dtype=weak.dtype);

        domTmp=self._copyToBem(vec,domTmp,self.op.domain);

        dualTmp=weak*domTmp;




        tmp=np.zeros(self.shape[0]);
        tmp=self._copyFromBem(dualTmp,tmp,self.op.dual_to_range);

        return tmp;

    def _matmat(self,vec):
        print('matmat not implemented')

class NgBlock(_LinearOperator):
    def __init__(self, blf, blf2=None):
        self.blf=blf;
        if(blf2==None):
            self.tmp1 = blf.mat.CreateColVector();
            self.tmp2 = blf.mat.CreateColVector();
        else:
            self.tmp1 = blf2.mat.CreateColVector();
            self.tmp2 = blf2.mat.CreateColVector();
        self.shape=[blf.mat.height,blf.mat.width]
        self.dtype=self.tmp1.FV().NumPy().dtype;

    def _matvec(self,v):
        import numpy as np
        self.tmp1.FV().NumPy()[:] = v.reshape(v.shape[0]);
        self.tmp2.data = self.blf.mat * self.tmp1
        return self.tmp2.FV().NumPy()
    def _matmat(self,vec):
        print('matmat not implemented')
