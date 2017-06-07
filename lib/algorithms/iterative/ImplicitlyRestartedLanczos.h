    /*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./lib/algorithms/iterative/ImplicitlyRestartedLanczos.h

    Copyright (C) 2015

Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: paboyle <paboyle@ph.ed.ac.uk>
Author: Chulwoo Jung <chulwoo@bnl.gov>
Author: Christoph Lehner <clehner@bnl.gov>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
    *************************************************************************************/
    /*  END LEGAL */
#ifndef GRID_IRL_H
#define GRID_IRL_H

#include <Grid/Eigen/Dense>

#define USE_LAPACK_IRL

#include <string.h> //memset
#ifdef USE_LAPACK_IRL
extern "C" void dstegr(char *jobz, char *range, int *n, double *d, double *e,
		       double *vl, double *vu, int *il, int *iu, double *abstol,
		       int *m, double *w, double *z, int *ldz, int *isuppz,
		       double *work, int *lwork, int *iwork, int *liwork,
		       int *info);
#endif

#include <Grid/algorithms/densematrix/DenseMatrix.h>
#include <Grid/algorithms/iterative/EigenSort.h>
#include <zlib.h>
#include <sys/stat.h>
// eliminate temorary vector in calc()

namespace Grid {

template<typename Field>
class BlockedGrid {
public:
  GridBase* _grid;
  typedef typename Field::scalar_type Coeff_t;
  typedef typename Field::vector_type vCoeff_t;
  
  std::vector<int> _bs; // block size
  std::vector<int> _nb; // number of blocks
  std::vector<int> _l;  // local dimensions irrespective of cb
  std::vector<int> _l_cb;  // local dimensions of checkerboarded vector
  std::vector<int> _l_cb_o;  // local dimensions of inner checkerboarded vector
  std::vector<int> _bs_cb; // block size in checkerboarded vector
  std::vector<int> _nb_o; // number of blocks of simd o-sites

  int _nd, _blocks, _cf_size, _cf_block_size, _cf_o_block_size, _o_blocks, _block_sites;
  
  BlockedGrid(GridBase* grid, const std::vector<int>& block_size) :
    _grid(grid), _bs(block_size), _nd((int)_bs.size()), 
      _nb(block_size), _l(block_size), _l_cb(block_size), _nb_o(block_size),
      _l_cb_o(block_size), _bs_cb(block_size) {

    _blocks = 1;
    _o_blocks = 1;
    _l = grid->FullDimensions();
    _l_cb = grid->LocalDimensions();
    _l_cb_o = grid->_rdimensions;

    _cf_size = 1;
    _block_sites = 1;
    for (int i=0;i<_nd;i++) {
      _l[i] /= grid->_processors[i];

      assert(!(_l[i] % _bs[i])); // lattice must accommodate choice of blocksize

      int r = _l[i] / _l_cb[i];
      assert(!(_bs[i] % r)); // checkerboarding must accommodate choice of blocksize
      _bs_cb[i] = _bs[i] / r;
      _block_sites *= _bs_cb[i];
      _nb[i] = _l[i] / _bs[i];
      _nb_o[i] = _nb[i] / _grid->_simd_layout[i];
      assert(!(_nb[i] % _grid->_simd_layout[i])); // simd must accommodate choice of blocksize
      _blocks *= _nb[i];
      _o_blocks *= _nb_o[i];
      _cf_size *= _l[i];
    }

    _cf_size *= 12 / 2;
    _cf_block_size = _cf_size / _blocks;
    _cf_o_block_size = _cf_size / _o_blocks;

    std::cout << GridLogMessage << "BlockedGrid:" << std::endl;
    std::cout << GridLogMessage << " _l     = " << _l << std::endl;
    std::cout << GridLogMessage << " _l_cb     = " << _l_cb << std::endl;
    std::cout << GridLogMessage << " _l_cb_o     = " << _l_cb_o << std::endl;
    std::cout << GridLogMessage << " _bs    = " << _bs << std::endl;
    std::cout << GridLogMessage << " _bs_cb    = " << _bs_cb << std::endl;

    std::cout << GridLogMessage << " _nb    = " << _nb << std::endl;
    std::cout << GridLogMessage << " _nb_o    = " << _nb_o << std::endl;
    std::cout << GridLogMessage << " _blocks = " << _blocks << std::endl;
    std::cout << GridLogMessage << " _o_blocks = " << _o_blocks << std::endl;
    std::cout << GridLogMessage << " sizeof(vCoeff_t) = " << sizeof(vCoeff_t) << std::endl;
    std::cout << GridLogMessage << " _cf_size = " << _cf_size << std::endl;
    std::cout << GridLogMessage << " _cf_block_size = " << _cf_block_size << std::endl;
    std::cout << GridLogMessage << " _block_sites = " << _block_sites << std::endl;
    std::cout << GridLogMessage << " _grid->oSites() = " << _grid->oSites() << std::endl;

    //    _grid->Barrier();
    //abort();
  }

    void block_to_coor(int b, std::vector<int>& x0) {

      std::vector<int> bcoor;
      bcoor.resize(_nd);
      x0.resize(_nd);
      assert(b < _o_blocks);
      Lexicographic::CoorFromIndex(bcoor,b,_nb_o);
      int i;

      for (i=0;i<_nd;i++) {
	x0[i] = bcoor[i]*_bs_cb[i];
      }

      //std::cout << GridLogMessage << "Map block b -> " << x0 << std::endl;

    }

    void block_site_to_o_coor(const std::vector<int>& x0, std::vector<int>& coor, int i) {
      Lexicographic::CoorFromIndex(coor,i,_bs_cb);
      for (int j=0;j<_nd;j++)
	coor[j] += x0[j];
    }

    int block_site_to_o_site(const std::vector<int>& x0, int i) {
      std::vector<int> coor;  coor.resize(_nd);
      block_site_to_o_coor(x0,coor,i);
      Lexicographic::IndexFromCoor(coor,i,_l_cb_o);
      return i;
    }

    vCoeff_t block_sp(int b, const Field& x, const Field& y) {

      std::vector<int> x0;
      block_to_coor(b,x0);

      vCoeff_t ret = 0.0;
      for (int i=0;i<_block_sites;i++) { // only odd sites
	int ss = block_site_to_o_site(x0,i);
	ret += TensorRemove(innerProduct(x._odata[ss],y._odata[ss]));
      }

      return ret;

    }

    vCoeff_t block_sp(int b, const Field& x, const std::vector< ComplexD >& y) {

      std::vector<int> x0;
      block_to_coor(b,x0);

      constexpr int nsimd = sizeof(vCoeff_t) / sizeof(Coeff_t);
      int lsize = _cf_o_block_size / _block_sites;

      std::vector< ComplexD > ret(nsimd);
      for (int i=0;i<nsimd;i++)
	ret[i] = 0.0;

      for (int i=0;i<_block_sites;i++) { // only odd sites
	int ss = block_site_to_o_site(x0,i);

	int n = lsize / nsimd;
	for (int l=0;l<n;l++) {
	  for (int j=0;j<nsimd;j++) {
	    int t = lsize * i + l*nsimd + j;

	    ret[j] += conjugate(((Coeff_t*)&x._odata[ss]._internal)[l*nsimd + j]) * y[t];
	  }
	}
      }

      vCoeff_t vret;
      for (int i=0;i<nsimd;i++)
	((Coeff_t*)&vret)[i] = (Coeff_t)ret[i];

      return vret;

    }

    template<class T>
      void vcaxpy(iScalar<T>& r,const vCoeff_t& a,const iScalar<T>& x,const iScalar<T>& y) {
      vcaxpy(r._internal,a,x._internal,y._internal);
    }

    template<class T,int N>
      void vcaxpy(iVector<T,N>& r,const vCoeff_t& a,const iVector<T,N>& x,const iVector<T,N>& y) {
      for (int i=0;i<N;i++)
	vcaxpy(r._internal[i],a,x._internal[i],y._internal[i]);
    }

    void vcaxpy(vCoeff_t& r,const vCoeff_t& a,const vCoeff_t& x,const vCoeff_t& y) {
      r = a*x + y;
    }

    void block_caxpy(int b, Field& ret, const vCoeff_t& a, const Field& x, const Field& y) {

      std::vector<int> x0;
      block_to_coor(b,x0);

      for (int i=0;i<_block_sites;i++) { // only odd sites
	int ss = block_site_to_o_site(x0,i);
	vcaxpy(ret._odata[ss],a,x._odata[ss],y._odata[ss]);
      }

    }

    void block_caxpy(int b, std::vector< ComplexD >& ret, const vCoeff_t& a, const Field& x, const std::vector< ComplexD >& y) {
      std::vector<int> x0;
      block_to_coor(b,x0);

      constexpr int nsimd = sizeof(vCoeff_t) / sizeof(Coeff_t);
      int lsize = _cf_o_block_size / _block_sites;

      for (int i=0;i<_block_sites;i++) { // only odd sites
	int ss = block_site_to_o_site(x0,i);

	int n = lsize / nsimd;
	for (int l=0;l<n;l++) {
	  vCoeff_t r = a* ((vCoeff_t*)&x._odata[ss]._internal)[l];

	  for (int j=0;j<nsimd;j++) {
	    int t = lsize * i + l*nsimd + j;
	    ret[t] = y[t] + ((Coeff_t*)&r)[j];
	  }
	}
      }

    }

    void block_set(int b, Field& ret, const std::vector< ComplexD >& x) {
      std::vector<int> x0;
      block_to_coor(b,x0);

      int lsize = _cf_o_block_size / _block_sites;

      for (int i=0;i<_block_sites;i++) { // only odd sites
	int ss = block_site_to_o_site(x0,i);

	for (int l=0;l<lsize;l++)
	  ((Coeff_t*)&ret._odata[ss]._internal)[l] = (Coeff_t)x[lsize * i + l]; // convert precision
      }

    }

    void block_get(int b, const Field& ret, std::vector< ComplexD >& x) {
      std::vector<int> x0;
      block_to_coor(b,x0);

      int lsize = _cf_o_block_size / _block_sites;

      for (int i=0;i<_block_sites;i++) { // only odd sites
	int ss = block_site_to_o_site(x0,i);

	for (int l=0;l<lsize;l++)
	  x[lsize * i + l] = (ComplexD)((Coeff_t*)&ret._odata[ss]._internal)[l];
      }

    }

    template<class T>
    void vcscale(iScalar<T>& r,const vCoeff_t& a,const iScalar<T>& x) {
      vcscale(r._internal,a,x._internal);
    }

    template<class T,int N>
    void vcscale(iVector<T,N>& r,const vCoeff_t& a,const iVector<T,N>& x) {
      for (int i=0;i<N;i++)
	vcscale(r._internal[i],a,x._internal[i]);
    }

    void vcscale(vCoeff_t& r,const vCoeff_t& a,const vCoeff_t& x) {
      r = a*x;
    }

    void block_cscale(int b, const vCoeff_t& a, Field& ret) {

      std::vector<int> x0;
      block_to_coor(b,x0);
      
      for (int i=0;i<_block_sites;i++) { // only odd sites
	int ss = block_site_to_o_site(x0,i);
	vcscale(ret._odata[ss],a,ret._odata[ss]);
      }
    }

    void getCanonicalBlockOffset(int cb, std::vector<int>& x0) {
      const int ndim = 5;
      assert(_nb.size() == ndim);
      std::vector<int> _nbc = { _nb[1], _nb[2], _nb[3], _nb[4], _nb[0] };
      std::vector<int> _bsc = { _bs[1], _bs[2], _bs[3], _bs[4], _bs[0] };
      x0.resize(ndim);

      assert(cb >= 0);
      assert(cb < _nbc[0]*_nbc[1]*_nbc[2]*_nbc[3]*_nbc[4]);

      Lexicographic::CoorFromIndex(x0,cb,_nbc);
      int i;

      for (i=0;i<ndim;i++) {
	x0[i] *= _bsc[i];
      }

      //if (cb < 2)
      //	std::cout << GridLogMessage << "Map: " << cb << " To: " << x0 << std::endl;
    }

    void pokeBlockOfVectorCanonical(int cb,Field& v,const std::vector<float>& buf) {
      std::vector<int> _bsc = { _bs[1], _bs[2], _bs[3], _bs[4], _bs[0] };
      std::vector<int> ldim = v._grid->LocalDimensions();
      std::vector<int> cldim = { ldim[1], ldim[2], ldim[3], ldim[4], ldim[0] };
      const int _nbsc = _bs_cb[0]*_bs_cb[1]*_bs_cb[2]*_bs_cb[3]*_bs_cb[4];
      // take canonical block cb of v and put it in canonical ordering in buf
      std::vector<int> cx0;
      getCanonicalBlockOffset(cb,cx0);

#pragma omp parallel
      {
	std::vector<int> co0,cl0;
	co0=cx0; cl0=cx0;

#pragma omp for
	for (int i=0;i<_nbsc;i++) {
	  Lexicographic::CoorFromIndex(co0,2*i,_bsc); // 2* for eo
	  for (int j=0;j<(int)_bsc.size();j++)
	    cl0[j] = cx0[j] + co0[j];
	  
	  std::vector<int> l0 = { cl0[4], cl0[0], cl0[1], cl0[2], cl0[3] };
	  int oi = v._grid->oIndex(l0);
	  int ii = v._grid->iIndex(l0);
	  int lti = i;

	  //if (cb < 2 && i<2)
	  //  std::cout << GridLogMessage << "Map: " << cb << ", " << i << " To: " << cl0 << ", " << cx0 << ", " << oi << ", " << ii << std::endl;
	  
	  for (int s=0;s<4;s++)
	    for (int c=0;c<3;c++) {
	      Coeff_t& ld = ((Coeff_t*)&v._odata[oi]._internal._internal[s]._internal[c])[ii];
	      int ti = 12*lti + 3*s + c;
	      ld = Coeff_t(buf[2*ti+0], buf[2*ti+1]);
	    }
	}
      }
    }

    void peekBlockOfVectorCanonical(int cb,const Field& v,std::vector<float>& buf) {
      std::vector<int> _bsc = { _bs[1], _bs[2], _bs[3], _bs[4], _bs[0] };
      std::vector<int> ldim = v._grid->LocalDimensions();
      std::vector<int> cldim = { ldim[1], ldim[2], ldim[3], ldim[4], ldim[0] };
      const int _nbsc = _bs_cb[0]*_bs_cb[1]*_bs_cb[2]*_bs_cb[3]*_bs_cb[4];
      // take canonical block cb of v and put it in canonical ordering in buf
      std::vector<int> cx0;
      getCanonicalBlockOffset(cb,cx0);

      buf.resize(_cf_block_size * 2);

#pragma omp parallel
      {
	std::vector<int> co0,cl0;
	co0=cx0; cl0=cx0;

#pragma omp for
	for (int i=0;i<_nbsc;i++) {
	  Lexicographic::CoorFromIndex(co0,2*i,_bsc); // 2* for eo
	  for (int j=0;j<(int)_bsc.size();j++)
	    cl0[j] = cx0[j] + co0[j];
	  
	  std::vector<int> l0 = { cl0[4], cl0[0], cl0[1], cl0[2], cl0[3] };
	  int oi = v._grid->oIndex(l0);
	  int ii = v._grid->iIndex(l0);
	  int lti = i;
	  
	  //if (cb < 2 && i<2)
	  //  std::cout << GridLogMessage << "Map: " << cb << ", " << i << " To: " << cl0 << ", " << cx0 << ", " << oi << ", " << ii << std::endl;

	  for (int s=0;s<4;s++)
	    for (int c=0;c<3;c++) {
	      Coeff_t& ld = ((Coeff_t*)&v._odata[oi]._internal._internal[s]._internal[c])[ii];
	      int ti = 12*lti + 3*s + c;
	      buf[2*ti+0] = ld.real();
	      buf[2*ti+1] = ld.imag();
	    }
	}
      }
    }

    int globalToLocalCanonicalBlock(int slot,const std::vector<int>& src_nodes,int nb) {
      // processor coordinate
      int _nd = (int)src_nodes.size();
      std::vector<int> _src_nodes = src_nodes;
      std::vector<int> pco(_nd);
      Lexicographic::CoorFromIndex(pco,slot,_src_nodes);
      std::vector<int> cpco = { pco[1], pco[2], pco[3], pco[4], pco[0] };

      // get local block
      std::vector<int> _nbc = { _nb[1], _nb[2], _nb[3], _nb[4], _nb[0] };
      assert(_nd == 5);
      std::vector<int> c_src_local_blocks(_nd);
      for (int i=0;i<_nd;i++) {
	assert(_grid->_fdimensions[i] % (src_nodes[i] * _bs[i]) == 0);
	c_src_local_blocks[(i+4) % 5] = _grid->_fdimensions[i] / src_nodes[i] / _bs[i];
      }
      std::vector<int> cbcoor(_nd); // coordinate of block in slot in canonical form
      Lexicographic::CoorFromIndex(cbcoor,nb,c_src_local_blocks);

      // cpco, cbcoor
      std::vector<int> clbcoor(_nd);
      for (int i=0;i<_nd;i++) {
	int cgcoor = cpco[i] * c_src_local_blocks[i] + cbcoor[i]; // global block coordinate
	int pcoor = cgcoor / _nbc[i]; // processor coordinate in my Grid
	int tpcoor = _grid->_processor_coor[(i+1)%5];
	if (pcoor != tpcoor)
	  return -1;
	clbcoor[i] = cgcoor - tpcoor * _nbc[i]; // canonical local block coordinate for canonical dimension i
      }

      int lnb;
      Lexicographic::IndexFromCoor(clbcoor,lnb,_nbc);
      //std::cout << "Mapped slot = " << slot << " nb = " << nb << " to " << lnb << std::endl;
      return lnb;
    }


 };



template<class Field>
class BasisFieldVector {
 public:
  int _Nm;

  typedef typename Field::scalar_type Coeff_t;
  typedef typename Field::vector_type vCoeff_t;
  typedef typename Field::vector_object vobj;
  typedef typename vobj::scalar_object sobj;

  std::vector<Field> _v; // _Nfull vectors
  
  BasisFieldVector(int Nm,GridBase* value) : _Nm(Nm), _v(Nm,value) {

    std::cout << GridLogMessage << "BasisFieldVector initialized:\n";
    std::cout << GridLogMessage << " Nm = " << Nm << "\n";
    std::cout << GridLogMessage << " Size of full vectors = " << 
      ((double)_v.size()*sizeof(vobj)*value->oSites() / 1024./1024./1024.) << " GB\n";
  }
  
  ~BasisFieldVector() {
  }

  Field& operator[](int i) {
    return _v[i];
  }

  void orthogonalize(Field& w, int k) {
    for(int j=0; j<k; ++j){
      Coeff_t ip = (Coeff_t)innerProduct(_v[j],w);
      w = w - ip*_v[j];
    }
  }

  void rotate(DenseVector<RealD>& Qt,int j0, int j1, int k0,int k1,int Nm) {
    
    GridBase* grid = _v[0]._grid;
      
#pragma omp parallel
    {
      std::vector < vobj > B(Nm);
      
#pragma omp for
      for(int ss=0;ss < grid->oSites();ss++){
	for(int j=j0; j<j1; ++j) B[j]=0.;
	
	for(int j=j0; j<j1; ++j){
	  for(int k=k0; k<k1; ++k){
	    B[j] +=Qt[k+Nm*j] * _v[k]._odata[ss];
	  }
	}
	for(int j=j0; j<j1; ++j){
	  _v[j]._odata[ss] = B[j];
	}
      }
    }

  }

  size_t size() const {
    return _Nm;
  }

  void resize(int n) {
    _Nm = n;
    _v.resize(n,_v[0]._grid);
  }

  std::vector<int> getIndex(DenseVector<RealD>& sort_vals) {

    std::vector<int> idx(sort_vals.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
	 [&sort_vals](int i1, int i2) {return ::fabs(sort_vals[i1]) < ::fabs(sort_vals[i2]);});

    return idx;
  }

  void reorderInPlace(DenseVector<RealD>& sort_vals, std::vector<int>& idx) {
    GridStopWatch gsw;
    gsw.Start();

    int nswaps = 0;
    for (size_t i=0;i<idx.size();i++) {
      if (idx[i] != i) {

	// find proper place (this could be done in logarithmic time, don't bother for now)
	size_t j;
	for (j=i;j<idx.size();j++)
	  if (idx[j]==i)
	    break;
	assert(j!=idx.size());
	
	Field _t(_v[0]._grid);
	_t = _v[idx[j]];
	_v[idx[j]] = _v[idx[i]];
	_v[idx[i]] = _t;

	RealD _td = sort_vals[idx[j]];
	sort_vals[idx[j]] = sort_vals[idx[i]];
	sort_vals[idx[i]] = _td;

	int _tt = idx[i];
	idx[i] = idx[j];
	idx[j] = _tt;
	
	nswaps++;
      }
    }

    // sort values
    gsw.Stop();
    std::cout << GridLogMessage << "Sorted eigenspace in place in " << gsw.Elapsed() << " using " << nswaps << " swaps" << std::endl;
  }

  void sortInPlace(DenseVector<RealD>& sort_vals, bool reverse) {

    std::vector<int> idx = getIndex(sort_vals);
    if (reverse)
      std::reverse(idx.begin(), idx.end());

    reorderInPlace(sort_vals,idx);

  }

 }; 

/*
  BlockProjector

  If _HP_BLOCK_PROJECTORS_ is defined, we assume that _evec is a basis that is not
  fully orthonormalized (to the precision of the coarse field) and we allow for higher-precision
  coarse field than basis field.

*/
//#define _HP_BLOCK_PROJECTORS_

template<typename Field>
class BlockProjector {
public:

  BasisFieldVector<Field>& _evec;
  BlockedGrid<Field>& _bgrid;

  BlockProjector(BasisFieldVector<Field>& evec, BlockedGrid<Field>& bgrid) : _evec(evec), _bgrid(bgrid) {
  }

  void createOrthonormalBasis(RealD thres = 0.0) {

    GridStopWatch sw;
    sw.Start();

    int cnt = 0;

#pragma omp parallel shared(cnt)
    {
      int lcnt = 0;

#pragma omp for
      for (int b=0;b<_bgrid._o_blocks;b++) {
	
	for (int i=0;i<_evec._Nm;i++) {
	  
	  auto nrm0 = _bgrid.block_sp(b,_evec._v[i],_evec._v[i]);
	  
	  // |i> -= <j|i> |j>
	  for (int j=0;j<i;j++) {
	    _bgrid.block_caxpy(b,_evec._v[i],-_bgrid.block_sp(b,_evec._v[j],_evec._v[i]),_evec._v[j],_evec._v[i]);
	  }
	  
	  auto nrm = _bgrid.block_sp(b,_evec._v[i],_evec._v[i]);
	  
	  auto eps = nrm/nrm0;
	  if (Reduce(eps).real() < thres) {
	    lcnt++;
	  }
	  
	  // TODO: if norm is too small, remove this eigenvector/mark as not needed; in practice: set it to zero norm here and return a mask
	  // that is then used later to decide not to write certain eigenvectors to disk (add a norm calculation before subtraction step and look at nrm/nrm0 < eps to decide)
	  _bgrid.block_cscale(b,1.0 / sqrt(nrm),_evec._v[i]);
	  
	}
	
      }

#pragma omp critical
      {
	cnt += lcnt;
      }
    }
    sw.Stop();
    std::cout << GridLogMessage << "Gram-Schmidt to create blocked basis took " << sw.Elapsed() << " (" << ((RealD)cnt / (RealD)_bgrid._o_blocks / (RealD)_evec._Nm) 
	      << " below threshold)" << std::endl;

  }

  template<typename CoarseField>
  void coarseToFine(const CoarseField& in, Field& out) {

    out = zero;
    out.checkerboard = _evec._v[0].checkerboard;

    int Nbasis = sizeof(in._odata[0]._internal._internal) / sizeof(in._odata[0]._internal._internal[0]);
    assert(Nbasis == _evec._Nm);
    
#pragma omp parallel for
    for (int b=0;b<_bgrid._o_blocks;b++) {
      for (int j=0;j<_evec._Nm;j++) {
	_bgrid.block_caxpy(b,out,in._odata[b]._internal._internal[j],_evec._v[j],out);
      }
    }

  }

  template<typename CoarseField>
  void fineToCoarse(const Field& in, CoarseField& out) {

    out = zero;

    int Nbasis = sizeof(out._odata[0]._internal._internal) / sizeof(out._odata[0]._internal._internal[0]);
    assert(Nbasis == _evec._Nm);


    Field tmp(_bgrid._grid);
    tmp = in;
    
#pragma omp parallel for
    for (int b=0;b<_bgrid._o_blocks;b++) {
      for (int j=0;j<_evec._Nm;j++) {
	// |rhs> -= <j|rhs> |j>
	auto c = _bgrid.block_sp(b,_evec._v[j],tmp);
	_bgrid.block_caxpy(b,tmp,-c,_evec._v[j],tmp); // may make this more numerically stable
	out._odata[b]._internal._internal[j] = c;
      }
    }

  }

  template<typename CoarseField>
    void deflateFine(BasisFieldVector<CoarseField>& _coef,const std::vector<RealD>& eval,int N,const Field& src_orig,Field& result) {
    result = zero;
    for (int i=0;i<N;i++) {
      Field tmp(result._grid);
      coarseToFine(_coef._v[i],tmp);
      axpy(result,TensorRemove(innerProduct(tmp,src_orig)) / eval[i],tmp,result);
    }
  }

  template<typename CoarseField>
    void deflateCoarse(BasisFieldVector<CoarseField>& _coef,const std::vector<RealD>& eval,int N,const Field& src_orig,Field& result) {
    CoarseField src_coarse(_coef._v[0]._grid);
    CoarseField result_coarse = src_coarse;
    result_coarse = zero;
    fineToCoarse(src_orig,src_coarse);
    for (int i=0;i<N;i++) {
      axpy(result_coarse,TensorRemove(innerProduct(_coef._v[i],src_coarse)) / eval[i],_coef._v[i],result_coarse);
    }
    coarseToFine(result_coarse,result);
  }

  template<typename CoarseField>
    void deflate(BasisFieldVector<CoarseField>& _coef,const std::vector<RealD>& eval,int N,const Field& src_orig,Field& result) {
    // Deflation on coarse Grid is much faster, so use it by default.  Deflation on fine Grid is kept for legacy reasons for now.
    deflateCoarse(_coef,eval,N,src_orig,result);
  }

};



 namespace FieldVectorIO {

   // zlib's crc32 gets 0.4 GB/s on KNL single thread
   // below gets 4.8 GB/s
   static uint32_t crc32_threaded(unsigned char* data, int64_t len, uint32_t previousCrc32 = 0) {

     // crc32 of zlib was incorrect for very large sizes, so do it block-wise
     uint32_t crc = previousCrc32;
     off_t blk = 0;
     off_t step = 1024*1024*1024;
     while (len > step) {
       crc = crc32(crc,&data[blk],step);
       blk += step;
       len -= step;
     }

     crc = crc32(crc,&data[blk],len);
     return crc;

   }

   static int get_bfm_index( int* pos, int co, int* s ) {
     
     int ls = s[0];
     int NtHalf = s[4] / 2;
     int simd_coor = pos[4] / NtHalf;
     int regu_coor = (pos[1] + s[1] * (pos[2] + s[2] * ( pos[3] + s[3] * (pos[4] % NtHalf) ) )) / 2;
     
     return regu_coor * ls * 48 + pos[0] * 48 + co * 4 + simd_coor * 2;
   }
   
   static void get_read_geometry(const GridBase* _grid,const std::vector<int>& cnodes,
			  std::map<int, std::vector<int> >& slots, 
			  std::vector<int>& slot_lvol,
			  std::vector<int>& lvol,
			  int64_t& slot_lsites,int& ntotal) {

     int _nd = (int)cnodes.size();
     std::vector<int> nodes = cnodes;

     slots.clear();
     slot_lvol.clear();
     lvol.clear();

     int i;
     ntotal = 1;
     int64_t lsites = 1;
     slot_lsites = 1;
     for (i=0;i<_nd;i++) {
       assert(_grid->_fdimensions[i] % nodes[i] == 0);
       slot_lvol.push_back(_grid->_fdimensions[i] / nodes[i]);
       lvol.push_back(_grid->_fdimensions[i] / _grid->_processors[i]);
       lsites *= lvol.back();
       slot_lsites *= slot_lvol.back();
       ntotal *= nodes[i];
     }

     std::vector<int> lcoor, gcoor, scoor;
     lcoor.resize(_nd); gcoor.resize(_nd);  scoor.resize(_nd);
     
     // create mapping of indices to slots
     for (int lidx = 0; lidx < lsites; lidx++) {
       Lexicographic::CoorFromIndex(lcoor,lidx,lvol);
       for (int i=0;i<_nd;i++) {
	 gcoor[i] = lcoor[i] + _grid->_processor_coor[i]*lvol[i];
	 scoor[i] = gcoor[i] / slot_lvol[i];
       }
       int slot;
       Lexicographic::IndexFromCoor(scoor,slot,nodes);
       auto sl = slots.find(slot);
       if (sl == slots.end())
	 slots[slot] = std::vector<int>();
       slots[slot].push_back(lidx);
     }
   }
   
   static void canonical_block_to_coarse_coordinates(GridBase* _coarsegrid,int nb,int& ii,int& oi) {
      // canonical nb needs to be mapped in a coordinate on my coarsegrid (ii,io)
      std::vector<int> _l = _coarsegrid->LocalDimensions();
      std::vector<int> _cl = { _l[1], _l[2], _l[3], _l[4], _l[0] };
      std::vector<int> _cc(_l.size());
      Lexicographic::CoorFromIndex(_cc,nb,_cl);
      std::vector<int> _c = { _cc[4], _cc[0], _cc[1], _cc[2], _cc[3] };
      ii = _coarsegrid->iIndex(_c);
      oi = _coarsegrid->oIndex(_c);
    }

    template<typename Field>
     static bool read_argonne(BasisFieldVector<Field>& ret,const char* dir, const std::vector<int>& cnodes) {

     GridBase* _grid = ret._v[0]._grid;

     std::map<int, std::vector<int> > slots;
     std::vector<int> slot_lvol, lvol;
     int64_t slot_lsites;
     int ntotal;
     get_read_geometry(_grid,cnodes,
		       slots,slot_lvol,lvol,slot_lsites,
		       ntotal);
     int _nd = (int)lvol.size();

     // this is slow code to read the argonne file format for debugging purposes
     int nperdir = ntotal / 32;
     if (nperdir < 1)
       nperdir=1;
     std::cout << GridLogMessage << " Read " << dir << " nodes = " << cnodes << std::endl;
     std::cout << GridLogMessage << " lvol = " << lvol << std::endl;
     
     // for error messages
     char hostname[1024];
     gethostname(hostname, 1024);

     // now load one slot at a time and fill the vector
     for (auto sl=slots.begin();sl!=slots.end();sl++) {
       std::vector<int>& idx = sl->second;
       int slot = sl->first;
       std::vector<float> rdata;
       
       char buf[4096];
       
       sprintf(buf,"%s/checksums.txt",dir);
       FILE* f = fopen(buf,"rt");
       if (!f) {
	 fprintf(stderr,"Node %s cannot read %s\n",hostname,buf); fflush(stderr);
	 return false;
       }
       
       for (int l=0;l<3+slot;l++)
	 fgets(buf,sizeof(buf),f);
       uint32_t crc_exp = strtol(buf, NULL, 16);
       fclose(f);
       
       // load one slot vector
       sprintf(buf,"%s/%2.2d/%10.10d",dir,slot/nperdir,slot);
       f = fopen(buf,"rb");
       if (!f) {
	 fprintf(stderr,"Node %s cannot read %s\n",hostname,buf); fflush(stderr);
	 return false;
       }
       
       fseeko(f,0,SEEK_END);
       off_t total_size = ftello(f);
       fseeko(f,0,SEEK_SET);
       
       int64_t size = slot_lsites / 2 * 24*4;
       rdata.resize(size);
       
       assert(total_size % size == 0);
       
       int _Nfull = total_size / size;
       ret._v.resize(_Nfull,ret._v[0]);
       ret._Nm = _Nfull;
       
       uint32_t crc = 0x0;
       GridStopWatch gsw,gsw2;
       for (int nev = 0;nev < _Nfull;nev++) {
	 
	 gsw.Start();
	 assert(fread(&rdata[0],size,1,f) == 1);
	 gsw.Stop();
	 
	 gsw2.Start();
	 crc = crc32_threaded((unsigned char*)&rdata[0],size,crc);
	 gsw2.Stop();
	 
	 for (int i=0;i<size/4;i++) {
	   char* c = (char*)&rdata[i];
	   char tmp; int j;
	   for (j=0;j<2;j++) {
	     tmp = c[j]; c[j] = c[3-j]; c[3-j] = tmp;
	   }
	 }
	 
	 // loop
	 ret._v[nev].checkerboard = Odd;
#pragma omp parallel 
	 {
	   
	   std::vector<int> lcoor, gcoor, scoor, slcoor;
	   lcoor.resize(_nd); gcoor.resize(_nd); 
	   slcoor.resize(_nd); scoor.resize(_nd);
	   
#pragma omp for
	   for (int64_t lidx = 0; lidx < idx.size(); lidx++) {
	     int llidx = idx[lidx];
	     Lexicographic::CoorFromIndex(lcoor,llidx,lvol);
	     for (int i=0;i<_nd;i++) {
	       gcoor[i] = lcoor[i] + _grid->_processor_coor[i]*lvol[i];
	       scoor[i] = gcoor[i] / slot_lvol[i];
	       slcoor[i] = gcoor[i] - scoor[i]*slot_lvol[i];
	     }
	     
	     if ((lcoor[1]+lcoor[2]+lcoor[3]+lcoor[4]) % 2 == 1) {
	       // poke 
	       iScalar<iVector<iVector<ComplexF, 3>, 4> > sc;
	       for (int s=0;s<4;s++)
		 for (int c=0;c<3;c++)
		   sc()(s)(c) = *(std::complex<float>*)&rdata[get_bfm_index(&slcoor[0],c+s*3, &slot_lvol[0] )];
	       
	       pokeLocalSite(sc,ret._v[nev],lcoor);
	     }
	     
	   }
	 }
       }
       
       fclose(f);      
       std::cout << GridLogMessage << "Loading slot " << slot << " with " << idx.size() << " points and " 
		 << _Nfull << " vectors in "
		 << gsw.Elapsed() << " at " 
		 << ( (double)size * _Nfull / 1024./1024./1024. / gsw.useconds()*1000.*1000. )
		 << " GB/s " << " crc32 = " << std::hex << crc << " crc32_expected = " << crc_exp << std::dec
		 << " computed at "
		 << ( (double)size * _Nfull / 1024./1024./1024. / gsw2.useconds()*1000.*1000. )
		 << " GB/s "
		 << std::endl;
       
       assert(crc == crc_exp);
     }

     _grid->Barrier();
     std::cout << GridLogMessage  << "Loading complete" << std::endl;
     
     return true;
   }

   template<typename Field>
     static bool read_argonne(BasisFieldVector<Field>& ret,const char* dir) {


     GridBase* _grid = ret._v[0]._grid;

     char buf[4096];
     sprintf(buf,"%s/nodes.txt",dir);
     FILE* f = fopen(buf,"rt");
     if (!f) {
       if (_grid->IsBoss()) {
	 fprintf(stderr,"Attempting to load eigenvectors without secifying node layout failed due to absence of nodes.txt\n");
	 fflush(stderr);
       }
       return false;
     }


     std::vector<int> nodes((int)_grid->_processors.size());
     for (int i =0;i<(int)_grid->_processors.size();i++)
       assert(fscanf(f,"%d\n",&nodes[i])==1);
     fclose(f);

     return read_argonne(ret,dir,nodes);
   }

   static void flush_bytes(FILE* f, std::vector<char>& fbuf) {
     if (fbuf.size()) {

       if (fwrite(&fbuf[0],fbuf.size(),1,f) != 1) {
	 fprintf(stderr,"Write failed of %g GB!\n",(double)fbuf.size() / 1024./1024./1024.);
	 exit(2);
       }

       fbuf.resize(0);

     }
   }

   static void write_bytes(void* buf, int64_t s, FILE* f, std::vector<char>& fbuf, uint32_t& crc) {
     static double data_counter = 0.0;
     static GridStopWatch gsw_crc, gsw_flush1,gsw_flush2,gsw_write,gsw_memcpy;
     if (s == 0)
       return;

     // checksum
     gsw_crc.Start();
     crc = crc32_threaded((unsigned char*)buf,s,crc);
     gsw_crc.Stop();

     if (s > fbuf.capacity()) {
       // cannot buffer this, so first flush current buffer contents and then write this directly to file
       gsw_flush1.Start();
       flush_bytes(f,fbuf);
       gsw_flush1.Stop();

       gsw_write.Start();
       if (fwrite(buf,s,1,f) != 1) {
	 fprintf(stderr,"Write failed of %g GB!\n",(double)s / 1024./1024./1024.);
	 exit(2);
       }
       gsw_write.Stop();

     }

     // no room left in buffer, flush to disk
     if (fbuf.size() + s > fbuf.capacity()) {
       gsw_flush2.Start();
       flush_bytes(f,fbuf);
       gsw_flush2.Stop();
     }

     // then fill buffer again
     {
       gsw_memcpy.Start();
       size_t t = fbuf.size();
       fbuf.resize(t + s);
       memcpy(&fbuf[t],buf,s);
       gsw_memcpy.Stop();
     }

     data_counter += (double)s;
     if (data_counter > 1024.*1024.*20.) {
       std::cout << GridLogMessage << "Writing " << ((double)data_counter / 1024./1024./1024.) << " GB at"
	 " crc = " << gsw_crc.Elapsed() << " flush1 = " << gsw_flush1.Elapsed() << " flush2 = " << gsw_flush2.Elapsed() <<
	 " write = " << gsw_write.Elapsed() << " memcpy = " << gsw_memcpy.Elapsed() << std::endl;
       data_counter = 0.0;
       gsw_crc.Reset();
       gsw_write.Reset();
       gsw_memcpy.Reset();
       gsw_flush1.Reset();
       gsw_flush2.Reset();
     }
   }

   static void write_floats(FILE* f, std::vector<char>& fbuf, uint32_t& crc, float* buf, int64_t n) {
     write_bytes(buf,n*sizeof(float),f,fbuf,crc);
   }

   static void read_floats(char* & ptr, float* out, int64_t n) {
     float* in = (float*)ptr;
     ptr += 4*n;

     for (int64_t i=0;i<n;i++)
       out[i] = in[i];
   }

   static int fp_map(float in, float min, float max, int N) {
     // Idea:
     //
     // min=-6
     // max=6
     //
     // N=1
     // [-6,0] -> 0, [0,6] -> 1;  reconstruct 0 -> -3, 1-> 3
     //
     // N=2
     // [-6,-2] -> 0, [-2,2] -> 1, [2,6] -> 2;  reconstruct 0 -> -4, 1->0, 2->4
     int ret =  (int) ( (float)(N+1) * ( (in - min) / (max - min) ) );
     if (ret == N+1) {
       ret = N;
     }
     return ret;
   }

   static float fp_unmap(int val, float min, float max, int N) {
     return min + (float)(val + 0.5) * (max - min)  / (float)( N + 1 );
   }

#define SHRT_UMAX 65535
#define FP16_BASE 1.4142135623730950488
#define FP16_COEF_EXP_SHARE_FLOATS 10
   static float unmap_fp16_exp(unsigned short e) {
     float de = (float)((int)e - SHRT_UMAX / 2);
     return ::pow( FP16_BASE, de );
   }

   // can assume that v >=0 and need to guarantee that unmap_fp16_exp(map_fp16_exp(v)) >= v
   static unsigned short map_fp16_exp(float v) {
     // float has exponents 10^{-44.85} .. 10^{38.53}
     int exp = (int)ceil(::log(v) / ::log(FP16_BASE)) + SHRT_UMAX / 2;
     if (exp < 0 || exp > SHRT_UMAX) {
       fprintf(stderr,"Error in map_fp16_exp(%g,%d)\n",v,exp);
       exit(3);
     }

     return (unsigned short)exp;
   }
  
   template<typename OPT>
     static void read_floats_fp16(char* & ptr, OPT* out, int64_t n, int nsc) {

     int64_t nsites = n / nsc;
     if (n % nsc) {
       fprintf(stderr,"Invalid size in write_floats_fp16\n");
       exit(4);
     }

     unsigned short* in = (unsigned short*)ptr;
     ptr += 2*(n+nsites);

     // do for each site
     for (int64_t site = 0;site<nsites;site++) {

       OPT* ev = &out[site*nsc];

       unsigned short* bptr = &in[site*(nsc + 1)];

       unsigned short exp = *bptr++;
       OPT max = unmap_fp16_exp(exp);
       OPT min = -max;

       for (int i=0;i<nsc;i++) {
	 ev[i] = fp_unmap( *bptr++, min, max, SHRT_UMAX );
       }

     }

   }

   template<typename OPT>
     static void write_floats_fp16(FILE* f, std::vector<char>& fbuf, uint32_t& crc, OPT* in, int64_t n, int nsc) {

     int64_t nsites = n / nsc;
     if (n % nsc) {
       fprintf(stderr,"Invalid size in write_floats_fp16\n");
       exit(4);
     }

     unsigned short* buf = (unsigned short*)malloc( sizeof(short) * (n + nsites) );
     if (!buf) {
       fprintf(stderr,"Out of mem\n");
       exit(1);
     }

     // do for each site
#pragma omp parallel for
     for (int64_t site = 0;site<nsites;site++) {

       OPT* ev = &in[site*nsc];

       unsigned short* bptr = &buf[site*(nsc + 1)];

       OPT max = fabs(ev[0]);
       OPT min;

       for (int i=0;i<nsc;i++) {
	 if (fabs(ev[i]) > max)
	   max = fabs(ev[i]);
       }

       unsigned short exp = map_fp16_exp(max);
       max = unmap_fp16_exp(exp);
       min = -max;

       *bptr++ = exp;

       for (int i=0;i<nsc;i++) {
	 int val = fp_map( ev[i], min, max, SHRT_UMAX );
	 if (val < 0 || val > SHRT_UMAX) {
	   fprintf(stderr,"Assert failed: val = %d (%d), ev[i] = %.15g, max = %.15g, exp = %d\n",val,SHRT_UMAX,ev[i],max,(int)exp);
	   exit(48);
	 }
	 *bptr++ = (unsigned short)val;
       }

     }

     write_bytes(buf,sizeof(short)*(n + nsites),f,fbuf,crc);

     free(buf);
   }

   template<typename Field,typename CoarseField>
     static bool read_compressed_vectors(const char* dir,BlockProjector<Field>& pr,BasisFieldVector<CoarseField>& coef) {

     const BasisFieldVector<Field>& basis = pr._evec;
     GridBase* _grid = basis._v[0]._grid;

     // for error messages
     char hostname[1024];
     gethostname(hostname, 1024);

     // first read metadata
     char buf[4096];
     sprintf(buf,"%s/metadata.txt",dir);

     std::vector<uint32_t> s,b,nb,nn,crc32;
     s.resize(5); b.resize(5); nb.resize(5); nn.resize(5);
     uint32_t neig, nkeep, nkeep_single, blocks, _FP16_COEF_EXP_SHARE_FLOATS;
     uint32_t nprocessors = 1;

     FILE* f = 0;
     uint32_t status = 0;
     if (_grid->IsBoss()) {
       f = fopen(buf,"rb");
       status=f ? 1 : 0;
     }
     _grid->GlobalSum(status);
     if (!status) {
       return false;
     }

#define _IRL_READ_INT(buf,p) if (f) { assert(fscanf(f,buf,p)==1); } else { *(p) = 0; } _grid->GlobalSum(*(p));

     for (int i=0;i<5;i++) {
       sprintf(buf,"s[%d] = %%d\n",i);
       _IRL_READ_INT(buf,&s[(i+1)%5]);
     }
     for (int i=0;i<5;i++) {
       sprintf(buf,"b[%d] = %%d\n",i);
       _IRL_READ_INT(buf,&b[(i+1)%5]);
     }
     for (int i=0;i<5;i++) {
       sprintf(buf,"nb[%d] = %%d\n",i);
       _IRL_READ_INT(buf,&nb[(i+1)%5]);
     }
     _IRL_READ_INT("neig = %d\n",&neig);
     _IRL_READ_INT("nkeep = %d\n",&nkeep);
     _IRL_READ_INT("nkeep_single = %d\n",&nkeep_single);
     _IRL_READ_INT("blocks = %d\n",&blocks);
     _IRL_READ_INT("FP16_COEF_EXP_SHARE_FLOATS = %d\n",&_FP16_COEF_EXP_SHARE_FLOATS);

     for (int i=0;i<5;i++) {
       assert(_grid->FullDimensions()[i] % s[i] == 0);
       nn[i] = _grid->FullDimensions()[i] / s[i];
       nprocessors *= nn[i];
     }

     std::cout << GridLogMessage << "Reading data that was generated on node-layout " << nn << std::endl;

     crc32.resize(nprocessors);
     for (int i =0;i<nprocessors;i++) {
       sprintf(buf,"crc32[%d] = %%X\n",i);
       _IRL_READ_INT(buf,&crc32[i]);
     }

#undef _IRL_READ_INT

     if (f)
       fclose(f);

     // allocate memory
     assert(std::equal(pr._bgrid._bs.begin(),pr._bgrid._bs.end(),b.begin())); // needs to be the same for now
     assert(pr._evec.size() == nkeep); // needs to be the same since this is compile-time fixed
     coef.resize(neig);


     // now get read geometry
     std::map<int, std::vector<int> > slots;
     std::vector<int> slot_lvol, lvol;
     int64_t slot_lsites;
     int ntotal;
     std::vector<int> _nn(nn.begin(),nn.end());
     get_read_geometry(_grid,_nn,
		       slots,slot_lvol,lvol,slot_lsites,
		       ntotal);
     int _nd = (int)lvol.size();

     // types
     typedef typename Field::scalar_type Coeff_t;
     typedef typename CoarseField::scalar_type CoeffCoarse_t;

     // slot layout
     int nperdir = ntotal / 32;
     if (nperdir < 1)
       nperdir=1;

     // load all necessary slots and store them appropriately
     for (auto sl=slots.begin();sl!=slots.end();sl++) {
       std::vector<int>& idx = sl->second;
       int slot = sl->first;
       std::vector<float> rdata;
       
       char buf[4096];
       
       // load one slot vector
       sprintf(buf,"%s/%2.2d/%10.10d.compressed",dir,slot/nperdir,slot);
       f = fopen(buf,"rb");
       if (!f) {
	 fprintf(stderr,"Node %s cannot read %s\n",hostname,buf); fflush(stderr);
	 return false;
       }

       uint32_t crc = 0x0;
       off_t size;

       GridStopWatch gsw;
       _grid->Barrier();
       gsw.Start();

       fseeko(f,0,SEEK_END);
       size = ftello(f);
       fseeko(f,0,SEEK_SET);

       std::vector<char> raw_in(size);
       assert(fread(&raw_in[0],size,1,f) == 1);

       _grid->Barrier();
       gsw.Stop();

       RealD totalGB = (RealD)size / 1024./1024./1024 * _grid->_Nprocessors;
       RealD seconds = gsw.useconds() / 1e6;
       std::cout << GridLogMessage << "[" << slot << "]  Read " << totalGB << " GB of compressed data at " << totalGB/seconds << " GB/s" << std::endl;

       uint32_t crc_comp = crc32_threaded((unsigned char*)&raw_in[0],size,0);

       if (crc_comp != crc32[slot]) {
	 std::cout << "Node " << hostname << " found crc mismatch for file " << buf << " (" << std::hex << crc_comp << " vs " << crc32[slot] << std::dec << ")" << std::endl;
	 std::cout << "Byte size: " << size << std::endl;
       }

       _grid->Barrier();
       
       assert(crc_comp == crc32[slot]);
    
       fclose(f);

       char* ptr = &raw_in[0];

       GridStopWatch gsw2;
       gsw2.Start();
       {
	 int nsingleCap = nkeep_single;
	 if (pr._evec.size() < nsingleCap)
	   nsingleCap = pr._evec.size();

	 int _cf_block_size = slot_lsites * 12 / 2 / blocks;

#define FP_16_SIZE(a,b)  (( (a) + (a/b) )*2)
	 
	 // first read single precision basis vectors
#pragma omp parallel
	 {
	   std::vector<float> buf(_cf_block_size * 2);
#pragma omp for
	   for (int nb=0;nb<blocks;nb++) {
	     for (int i=0;i<nsingleCap;i++) {
               char* lptr = ptr + buf.size()*(i + nsingleCap*nb)*4;
	       read_floats(lptr, &buf[0], buf.size() );
	       int mnb = pr._bgrid.globalToLocalCanonicalBlock(slot,_nn,nb);
	       if (mnb != -1)
		 pr._bgrid.pokeBlockOfVectorCanonical(mnb,pr._evec._v[i],buf);
	     }
	   }

#pragma omp barrier
#pragma omp single
	   {
	     ptr = ptr + buf.size()*nsingleCap*blocks*4;
	   }

	 }

	 // TODO: at this point I should add a checksum test for block_sp(nb,v,v) for all blocks, then I would know that the mapping
	 // to blocks is OK at this point; after that ...
	 
	 // then read fixed precision basis vectors
#pragma omp parallel
	 {
	   std::vector<float> buf(_cf_block_size * 2);
#pragma omp for
	   for (int nb=0;nb<blocks;nb++) {
	     for (int i=nsingleCap;i<(int)pr._evec.size();i++) {
	       char* lptr = ptr + FP_16_SIZE( buf.size(), 24 )*((i-nsingleCap) + (pr._evec.size() - nsingleCap)*nb);
	       read_floats_fp16(lptr, &buf[0], buf.size(), 24);
	       int mnb = pr._bgrid.globalToLocalCanonicalBlock(slot,_nn,nb);
	       if (mnb != -1)
		 pr._bgrid.pokeBlockOfVectorCanonical(mnb,pr._evec._v[i],buf);
	     }
	   }

#pragma omp barrier
#pragma omp single
	   {
	     ptr = ptr + FP_16_SIZE( buf.size()*(pr._evec.size() - nsingleCap)*blocks, 24 );
	   }
	 }
	 
#pragma omp parallel
	 {
	   std::vector<float> buf1(nkeep_single*2);
	   std::vector<float> buf2((nkeep - nkeep_single)*2);

#pragma omp for
	   for (int j=0;j<(int)coef.size();j++)
	     for (int nb=0;nb<blocks;nb++) {
	       // get local coordinate on coarse grid
	       int ii,oi;
	       int mnb = pr._bgrid.globalToLocalCanonicalBlock(slot,_nn,nb);
	       if (mnb != -1)
		 canonical_block_to_coarse_coordinates(coef._v[0]._grid,mnb,ii,oi);

	       char* lptr = ptr + (4*buf1.size() + FP_16_SIZE(buf2.size(), _FP16_COEF_EXP_SHARE_FLOATS))*(nb + j*blocks);
	       int l;
	       read_floats(lptr, &buf1[0], buf1.size() );
	       if (mnb != -1) {
		 for (l=0;l<nkeep_single;l++) {
		   ((CoeffCoarse_t*)&coef._v[j]._odata[oi]._internal._internal[l])[ii] = CoeffCoarse_t(buf1[2*l+0],buf1[2*l+1]);
		 }
	       }
	       read_floats_fp16(lptr, &buf2[0], buf2.size(), _FP16_COEF_EXP_SHARE_FLOATS);
	       if (mnb != -1) {
		 for (l=nkeep_single;l<nkeep;l++) {
		   ((CoeffCoarse_t*)&coef._v[j]._odata[oi]._internal._internal[l])[ii] = CoeffCoarse_t(buf2[2*(l-nkeep_single)+0],buf2[2*(l-nkeep_single)+1]);
		 }
	       }

	     }
          }
	 
       }

       // set checkerboard
       for (int i=0;i<(int)pr._evec.size();i++)
	 pr._evec._v[i].checkerboard = Odd;

       gsw2.Stop();
       seconds=gsw2.useconds()/1e6;
       std::cout << GridLogMessage << "Processed " << totalGB << " GB of compressed data at " << totalGB/seconds << " GB/s" << std::endl;
     }
#undef FP_16_SIZE
     return true;

   }

   static bool DirectoryExists(const char *path) {
     struct stat info;
     return ((stat( path, &info ) == 0) && (info.st_mode & S_IFDIR));
   }

   static void conditionalMkDir(const char* path) {
     if (!DirectoryExists(path))
       mkdir(path,ACCESSPERMS);
   }

   template<typename Field,typename CoarseField>
   static void write_compressed_vectors(const char* dir,const BlockProjector<Field>& pr,const BasisFieldVector<CoarseField>& coef,
					int nsingle,int writer_nodes = 0) {

     GridStopWatch gsw;
     
     const BasisFieldVector<Field>& basis = pr._evec;
     GridBase* _grid = basis._v[0]._grid;
     std::vector<int> _l = _grid->FullDimensions();
     for (int i=0;i<(int)_l.size();i++)
       _l[i] /= _grid->_processors[i];

     _grid->Barrier();
     gsw.Start();

     char buf[4096];
     
     // Making the directories is somewhat tricky.
     // If we run on a joint filesystem we would just 
     // have the boss create the directories and then
     // have a barrier.  We also want to be able to run
     // on local /scratch, so potentially all nodes need
     // to create their own directories.  So do the following
     // for now.
     for (int j=0;j<_grid->_Nprocessors;j++) {
       if (j == _grid->ThisRank()) {
	 conditionalMkDir(dir);
	 for (int i=0;i<32;i++) {
	   sprintf(buf,"%s/%2.2d",dir,i);
	   conditionalMkDir(buf);
	 }       
	 _grid->Barrier(); // make sure directories are ready
       }
     }
     

     typedef typename Field::scalar_type Coeff_t;
     typedef typename CoarseField::scalar_type CoeffCoarse_t;
     
     int nperdir = _grid->_Nprocessors / 32;
     if (nperdir < 1)
       nperdir=1;

     int slot;
     Lexicographic::IndexFromCoor(_grid->_processor_coor,slot,_grid->_processors);

     int64_t off = 0x0;
     uint32_t crc = 0x0;
     if (writer_nodes < 1)
       writer_nodes = _grid->_Nprocessors;
     int groups = _grid->_Nprocessors / writer_nodes;
     if (groups<1)
       groups = 1;

     std::cout << GridLogMessage << " Write " << dir << " nodes = " << writer_nodes << std::endl;

     for (int group=0;group<groups;group++) {
       _grid->Barrier();
       if (_grid->ThisRank() % groups == group) {
	 
	 sprintf(buf,"%s/%2.2d/%10.10d.compressed",dir,slot/nperdir,slot);
	 FILE* f = fopen(buf,"wb");
	 assert(f);

	 //buffer does not seem to help
	 //assert(!setvbuf ( f , NULL , _IOFBF , 1024*1024*2 ));
	 	 
	 int nsingleCap = nsingle;
	 if (pr._evec.size() < nsingleCap)
	   nsingleCap = pr._evec.size();
	 
	 GridStopWatch gsw1,gsw2,gsw3,gsw4,gsw5;

	 gsw1.Start();

	 std::vector<char> fbuf;
	 fbuf.reserve( 1024 * 1024 * 8 );

	 // first write single precision basis vectors
	 for (int nb=0;nb<pr._bgrid._blocks;nb++) {
	   for (int i=0;i<nsingleCap;i++) {
	     std::vector<float> buf;
	     pr._bgrid.peekBlockOfVectorCanonical(nb,pr._evec._v[i],buf);
	     
#if 0
	     {
	       RealD nrm = 0.0;
	       for (int j=0;j<(int)buf.size();j++)
		 nrm += buf[j]*buf[j];
	       std::cout << GridLogMessage << "Norm: " << nrm << std::endl;
	     }
#endif
	     write_floats(f,fbuf,crc, &buf[0], buf.size() );
	   }
	 }

	 gsw1.Stop();
	 gsw2.Start();

	 // then write fixed precision basis vectors
	 for (int nb=0;nb<pr._bgrid._blocks;nb++) {
	   for (int i=nsingleCap;i<(int)pr._evec.size();i++) {
	     std::vector<float> buf;
	     pr._bgrid.peekBlockOfVectorCanonical(nb,pr._evec._v[i],buf);
	     write_floats_fp16(f,fbuf,crc, &buf[0], buf.size(), 24);
	   }
	 }

	 gsw2.Stop();
	 assert(coef._v[0]._grid->_isites*coef._v[0]._grid->_osites == pr._bgrid._blocks);

	 gsw3.Start();
	 for (int j=0;j<(int)coef.size();j++) {

	   int64_t size1 = nsingleCap*2;
	   int64_t size2 = 2*(pr._evec.size()-nsingleCap);
	   int64_t size = size1;
	   if (size2>size)
	     size=size2;
	   std::vector<float> buf(size);

	   //RealD nrmTest = 0.0;
	   for (int nb=0;nb<pr._bgrid._blocks;nb++) {
	     // get local coordinate on coarse grid
	     int ii, oi;
	     canonical_block_to_coarse_coordinates(coef._v[0]._grid,nb,ii,oi);
	     
	     gsw4.Start();
	     gsw5.Start();
	     for (int l=0;l<nsingleCap;l++) {
	       auto res = ((CoeffCoarse_t*)&coef._v[j]._odata[oi]._internal._internal[l])[ii];
	       buf[2*l+0] = res.real();
	       buf[2*l+1] = res.imag();
	       //nrmTest += res.real() * res.real() + res.imag() * res.imag();
	     }
	     gsw5.Stop();
	     write_floats(f,fbuf,crc, &buf[0], size1 );
	     gsw4.Stop();
	     for (int l=nsingleCap;l<(int)pr._evec.size();l++) {
	       auto res = ((CoeffCoarse_t*)&coef._v[j]._odata[oi]._internal._internal[l])[ii];
	       buf[2*(l-nsingleCap)+0] = res.real();
	       buf[2*(l-nsingleCap)+1] = res.imag();
	       //nrmTest += res.real() * res.real() + res.imag() * res.imag();
	     }
	     write_floats_fp16(f,fbuf,crc, &buf[0], size2, FP16_COEF_EXP_SHARE_FLOATS);
	   }

	   //_grid->GlobalSum(nrmTest);
	   //std::cout << GridLogMessage << "Test norm: " << nrmTest << std::endl;
	 }
	 gsw3.Stop();

	 flush_bytes(f,fbuf);

	 off = ftello(f);	 
	 fclose(f);

	 std::cout<<GridLogMessage << "Timing: write single basis in " << gsw1.Elapsed() << " and fp16 in " << gsw2.Elapsed() << " and coefficients in " << gsw3.Elapsed() << " (" <<
	   gsw4.Elapsed() << ", " << gsw5.Elapsed() << ")" << std::endl;
       }
     }
	 
     _grid->Barrier();
     gsw.Stop();
     
     RealD totalGB = (RealD)off / 1024./1024./1024 * _grid->_Nprocessors;
     RealD seconds = gsw.useconds() / 1e6;
     std::cout << GridLogMessage << "Write " << totalGB << " GB of compressed data at " << totalGB/seconds << " GB/s in " << seconds << " s" << std::endl;

     // gather crcs
     std::vector<uint32_t> crcs(_grid->_Nprocessors);
     for (int i=0;i<_grid->_Nprocessors;i++) {
       crcs[i] = 0x0;
     }
     crcs[slot] = crc;
     for (int i=0;i<_grid->_Nprocessors;i++) {
       _grid->GlobalSum(crcs[i]);
     }
     
     if (_grid->IsBoss()) {
       sprintf(buf,"%s/metadata.txt",dir);
       FILE* f = fopen(buf,"wb");
       assert(f);
       for (int i=0;i<5;i++)
	 fprintf(f,"s[%d] = %d\n",i,_grid->FullDimensions()[(i+1)%5] / _grid->_processors[(i+1)%5]);
       for (int i=0;i<5;i++)
	 fprintf(f,"b[%d] = %d\n",i,pr._bgrid._bs[(i+1)%5]);
       for (int i=0;i<5;i++)
	 fprintf(f,"nb[%d] = %d\n",i,pr._bgrid._nb[(i+1)%5]);
       fprintf(f,"neig = %d\n",(int)coef.size());
       fprintf(f,"nkeep = %d\n",(int)pr._evec.size());
       fprintf(f,"nkeep_single = %d\n",nsingle);
       fprintf(f,"blocks = %d\n",pr._bgrid._blocks);
       fprintf(f,"FP16_COEF_EXP_SHARE_FLOATS = %d\n",FP16_COEF_EXP_SHARE_FLOATS);
       for (int i =0;i<_grid->_Nprocessors;i++)
	 fprintf(f,"crc32[%d] = %X\n",i,crcs[i]);
       fclose(f);
     }

   }
   
   template<typename Field>
     static void write_argonne(const BasisFieldVector<Field>& ret,const char* dir) {
     
     GridBase* _grid = ret._v[0]._grid;
     std::vector<int> _l = _grid->FullDimensions();
     for (int i=0;i<(int)_l.size();i++)
       _l[i] /= _grid->_processors[i];

     char buf[4096];
     
     if (_grid->IsBoss()) {
       mkdir(dir,ACCESSPERMS);
       
       for (int i=0;i<32;i++) {
	 sprintf(buf,"%s/%2.2d",dir,i);
	 mkdir(buf,ACCESSPERMS);
       }
     }
     
     _grid->Barrier(); // make sure directories are ready

     
     int nperdir = _grid->_Nprocessors / 32;
     if (nperdir < 1)
       nperdir=1;
     std::cout << GridLogMessage << " Write " << dir << " nodes = " << _grid->_Nprocessors << std::endl;
     
     int slot;
     Lexicographic::IndexFromCoor(_grid->_processor_coor,slot,_grid->_processors);
     //printf("Slot: %d <> %d\n",slot, _grid->ThisRank());
     
     sprintf(buf,"%s/%2.2d/%10.10d",dir,slot/nperdir,slot);
     FILE* f = fopen(buf,"wb");
     assert(f);
     
     int N = (int)ret._v.size();
     uint32_t crc = 0x0;
     int64_t cf_size = _grid->oSites()*_grid->iSites()*12;
     std::vector< float > rdata(cf_size*2);
     
     GridStopWatch gsw1,gsw2;
     
     for (int i=0;i<N;i++) {

       if (ret._v[i].checkerboard != Odd)
	 continue;

       // create buffer and put data in argonne format in there
       std::vector<int> coor(_l.size());
       for (coor[1] = 0;coor[1]<_l[1];coor[1]++) {
	 for (coor[2] = 0;coor[2]<_l[2];coor[2]++) {
	   for (coor[3] = 0;coor[3]<_l[3];coor[3]++) {
	     for (coor[4] = 0;coor[4]<_l[4];coor[4]++) {
	       for (coor[0] = 0;coor[0]<_l[0];coor[0]++) {
		 
		 if ((coor[1]+coor[2]+coor[3]+coor[4]) % 2 == 1) {
		   // peek
		   iScalar<iVector<iVector<ComplexF, 3>, 4> > sc;
		   peekLocalSite(sc,ret._v[i],coor);
		   for (int s=0;s<4;s++)
		     for (int c=0;c<3;c++)
		       *(std::complex<float>*)&rdata[get_bfm_index(&coor[0],c+s*3, &_l[0] )] = sc()(s)(c);
		 }
	       }
	     }
	   }
	 }
       }
       
       // endian flip
       for (int i=0;i<cf_size*2;i++) {
	 char* c = (char*)&rdata[i];
	 char tmp; int j;
	 for (j=0;j<2;j++) {
	   tmp = c[j]; c[j] = c[3-j]; c[3-j] = tmp;
	 }
       }
       
       // create crc of buffer
       gsw1.Start();
       crc = crc32_threaded((unsigned char*)&rdata[0],cf_size*2*4,crc);    
       gsw1.Stop();
       
       // write out
       gsw2.Start();
       assert(fwrite(&rdata[0],cf_size*2*4,1,f)==1);
       gsw2.Stop();
       
     }
     
     fclose(f);
     
     
     // gather crc's and write out
     std::vector<uint32_t> crcs(_grid->_Nprocessors);
     for (int i=0;i<_grid->_Nprocessors;i++) {
       crcs[i] = 0x0;
     }
     crcs[slot] = crc;
     for (int i=0;i<_grid->_Nprocessors;i++) {
       _grid->GlobalSum(crcs[i]);
     }
     
     if (_grid->IsBoss()) {
       sprintf(buf,"%s/checksums.txt",dir);
       FILE* f = fopen(buf,"wt");
       assert(f);
       fprintf(f,"00000000\n\n");
       for (int i =0;i<_grid->_Nprocessors;i++)
	 fprintf(f,"%X\n",crcs[i]);
       fclose(f);
       
       sprintf(buf,"%s/nodes.txt",dir);
       f = fopen(buf,"wt");
       assert(f);
       for (int i =0;i<(int)_grid->_processors.size();i++)
	 fprintf(f,"%d\n",_grid->_processors[i]);
       fclose(f);
     }
     
     
     std::cout << GridLogMessage << "Writing slot " << slot << " with "
	       << N << " vectors in "
	       << gsw2.Elapsed() << " at " 
	       << ( (double)cf_size*2*4 * N / 1024./1024./1024. / gsw2.useconds()*1000.*1000. )
	       << " GB/s  with crc computed at "
	       << ( (double)cf_size*2*4 * N / 1024./1024./1024. / gsw1.useconds()*1000.*1000. )
	       << " GB/s "
	       << std::endl;
     
     _grid->Barrier();
     std::cout << GridLogMessage  << "Writing complete" << std::endl;
     
   }
   
 }

/////////////////////////////////////////////////////////////
// Implicitly restarted lanczos
/////////////////////////////////////////////////////////////

 template<class Field> 
    class ImplicitlyRestartedLanczos {

    const RealD small = 1.0e-16;
public:       
    int lock;
    int get;
    int Niter;
    int converged;

    int Nminres; // Minimum number of restarts; only check for convergence after
    int Nstop;   // Number of evecs checked for convergence
    int Nk;      // Number of converged sought
    int Np;      // Np -- Number of spare vecs in kryloc space
    int Nm;      // Nm -- total number of vectors


    RealD OrthoTime;

    RealD eresid, betastp;
    SortEigen<Field> _sort;
    LinearFunction<Field> &_HermOp;
    LinearFunction<Field> &_HermOpTest;
    /////////////////////////
    // Constructor
    /////////////////////////

    ImplicitlyRestartedLanczos(
			       LinearFunction<Field> & HermOp,
			       LinearFunction<Field> & HermOpTest,
			       int _Nstop, // sought vecs
			       int _Nk, // sought vecs
			       int _Nm, // spare vecs
			       RealD _eresid, // resid in lmdue deficit 
			       RealD _betastp, // if beta(k) < betastp: converged
			       int _Niter, // Max iterations
			       int _Nminres) :
      _HermOp(HermOp),
      _HermOpTest(HermOpTest),
      Nstop(_Nstop),
      Nk(_Nk),
      Nm(_Nm),
      eresid(_eresid),
      betastp(_betastp),
      Niter(_Niter),
      Nminres(_Nminres)
    { 
      Np = Nm-Nk; assert(Np>0);
    };

    ImplicitlyRestartedLanczos(
			       LinearFunction<Field> & HermOp,
			       LinearFunction<Field> & HermOpTest,
			       int _Nk, // sought vecs
			       int _Nm, // spare vecs
			       RealD _eresid, // resid in lmdue deficit 
			       RealD _betastp, // if beta(k) < betastp: converged
			       int _Niter, // Max iterations
			       int _Nminres) : 
      _HermOp(HermOp),
      _HermOpTest(HermOpTest),
      Nstop(_Nk),
      Nk(_Nk),
      Nm(_Nm),
      eresid(_eresid),
      betastp(_betastp),
      Niter(_Niter),
      Nminres(_Nminres)
    { 
      Np = Nm-Nk; assert(Np>0);
    };


/* Saad PP. 195
1. Choose an initial vector v1 of 2-norm unity. Set β1 ≡ 0, v0 ≡ 0
2. For k = 1,2,...,m Do:
3. wk:=Avk−βkv_{k−1}      
4. αk:=(wk,vk)       // 
5. wk:=wk−αkvk       // wk orthog vk 
6. βk+1 := ∥wk∥2. If βk+1 = 0 then Stop
7. vk+1 := wk/βk+1
8. EndDo
 */
    void step(DenseVector<RealD>& lmd,
	      DenseVector<RealD>& lme, 
	      BasisFieldVector<Field>& evec,
	      Field& w,int Nm,int k)
    {
      assert( k< Nm );

      GridStopWatch gsw_op,gsw_o;

      Field& evec_k = evec[k];

      gsw_op.Start();
      _HermOp(evec_k,w);
      gsw_op.Stop();

      if(k>0){
	w -= lme[k-1] * evec[k-1];
      }    

      ComplexD zalph = innerProduct(evec_k,w); // 4. αk:=(wk,vk)
      RealD     alph = real(zalph);

      w = w - alph * evec_k;// 5. wk:=wk−αkvk

      RealD beta = normalise(w); // 6. βk+1 := ∥wk∥2. If βk+1 = 0 then Stop
                                 // 7. vk+1 := wk/βk+1

      std::cout<<GridLogMessage << "alpha[" << k << "] = " << zalph << " beta[" << k << "] = "<<beta<<std::endl;
      const RealD tiny = 1.0e-20;
      if ( beta < tiny ) { 
	std::cout<<GridLogMessage << " beta is tiny "<<beta<<std::endl;
     }
      lmd[k] = alph;
      lme[k]  = beta;

      gsw_o.Start();
      if (k>0) { 
	orthogonalize(w,evec,k); // orthonormalise
      }
      gsw_o.Stop();

      if(k < Nm-1) { 
	evec[k+1] = w;
      }

      std::cout << GridLogMessage << "Timing: operator=" << gsw_op.Elapsed() <<
	" orth=" << gsw_o.Elapsed() << std::endl;

    }

    void qr_decomp(DenseVector<RealD>& lmd,
		   DenseVector<RealD>& lme,
		   int Nk,
		   int Nm,
		   DenseVector<RealD>& Qt,
		   RealD Dsh, 
		   int kmin,
		   int kmax)
    {
      int k = kmin-1;
      RealD x;

      RealD Fden = 1.0/hypot(lmd[k]-Dsh,lme[k]);
      RealD c = ( lmd[k] -Dsh) *Fden;
      RealD s = -lme[k] *Fden;
      
      RealD tmpa1 = lmd[k];
      RealD tmpa2 = lmd[k+1];
      RealD tmpb  = lme[k];

      lmd[k]   = c*c*tmpa1 +s*s*tmpa2 -2.0*c*s*tmpb;
      lmd[k+1] = s*s*tmpa1 +c*c*tmpa2 +2.0*c*s*tmpb;
      lme[k]   = c*s*(tmpa1-tmpa2) +(c*c-s*s)*tmpb;
      x        =-s*lme[k+1];
      lme[k+1] = c*lme[k+1];
      
      for(int i=0; i<Nk; ++i){
	RealD Qtmp1 = Qt[i+Nm*k  ];
	RealD Qtmp2 = Qt[i+Nm*(k+1)];
	Qt[i+Nm*k    ] = c*Qtmp1 - s*Qtmp2;
	Qt[i+Nm*(k+1)] = s*Qtmp1 + c*Qtmp2; 
      }

      // Givens transformations
      for(int k = kmin; k < kmax-1; ++k){

	RealD Fden = 1.0/hypot(x,lme[k-1]);
	RealD c = lme[k-1]*Fden;
	RealD s = - x*Fden;
	
	RealD tmpa1 = lmd[k];
	RealD tmpa2 = lmd[k+1];
	RealD tmpb  = lme[k];

	lmd[k]   = c*c*tmpa1 +s*s*tmpa2 -2.0*c*s*tmpb;
	lmd[k+1] = s*s*tmpa1 +c*c*tmpa2 +2.0*c*s*tmpb;
	lme[k]   = c*s*(tmpa1-tmpa2) +(c*c-s*s)*tmpb;
	lme[k-1] = c*lme[k-1] -s*x;

	if(k != kmax-2){
	  x = -s*lme[k+1];
	  lme[k+1] = c*lme[k+1];
	}

	for(int i=0; i<Nk; ++i){
	  RealD Qtmp1 = Qt[i+Nm*k    ];
	  RealD Qtmp2 = Qt[i+Nm*(k+1)];
	  Qt[i+Nm*k    ] = c*Qtmp1 -s*Qtmp2;
	  Qt[i+Nm*(k+1)] = s*Qtmp1 +c*Qtmp2;
	}
      }
    }

#ifdef USE_LAPACK_IRL
#define LAPACK_INT int
//long long
    void diagonalize_lapack(DenseVector<RealD>& lmd,
			    DenseVector<RealD>& lme, 
			    int N1,
			    int N2,
			    DenseVector<RealD>& Qt,
			    GridBase *grid){

      std::cout << GridLogMessage << "diagonalize_lapack start\n";
      GridStopWatch gsw;

      const int size = Nm;
      //  tevals.resize(size);
      //  tevecs.resize(size);
      LAPACK_INT NN = N1;
      std::vector<double> evals_tmp(NN);
      std::vector<double> evec_tmp(NN*NN);
      memset(&evec_tmp[0],0,sizeof(double)*NN*NN);
      //  double AA[NN][NN];
      std::vector<double> DD(NN);
      std::vector<double> EE(NN);
      for (int i = 0; i< NN; i++)
	for (int j = i - 1; j <= i + 1; j++)
	  if ( j < NN && j >= 0 ) {
	    if (i==j) DD[i] = lmd[i];
	    if (i==j) evals_tmp[i] = lmd[i];
	    if (j==(i-1)) EE[j] = lme[j];
	  }
      LAPACK_INT evals_found;
      LAPACK_INT lwork = ( (18*NN) > (1+4*NN+NN*NN)? (18*NN):(1+4*NN+NN*NN)) ;
      LAPACK_INT liwork =  3+NN*10 ;
      std::vector<LAPACK_INT> iwork(liwork);
      std::vector<double> work(lwork);
      std::vector<LAPACK_INT> isuppz(2*NN);
      char jobz = 'V'; // calculate evals & evecs
      char range = 'I'; // calculate all evals
      //    char range = 'A'; // calculate all evals
      char uplo = 'U'; // refer to upper half of original matrix
      char compz = 'I'; // Compute eigenvectors of tridiagonal matrix
      std::vector<int> ifail(NN);
      LAPACK_INT info;
      //  int total = QMP_get_number_of_nodes();
      //  int node = QMP_get_node_number();
      //  GridBase *grid = evec[0]._grid;
      int total = grid->_Nprocessors;
      int node = grid->_processor;
      int interval = (NN/total)+1;
      double vl = 0.0, vu = 0.0;
      LAPACK_INT il = interval*node+1 , iu = interval*(node+1);
      if (iu > NN)  iu=NN;
      double tol = 0.0;
      if (1) {
	memset(&evals_tmp[0],0,sizeof(double)*NN);
	if ( il <= NN){
	  std::cout << GridLogMessage << "dstegr started" << std::endl; 
	  gsw.Start();
	  dstegr(&jobz, &range, &NN,
		 (double*)&DD[0], (double*)&EE[0],
		 &vl, &vu, &il, &iu, // these four are ignored if second parameteris 'A'
		 &tol, // tolerance
		 &evals_found, &evals_tmp[0], (double*)&evec_tmp[0], &NN,
		 &isuppz[0],
		 &work[0], &lwork, &iwork[0], &liwork,
		 &info);
	  gsw.Stop();
	  std::cout << GridLogMessage << "dstegr completed in " << gsw.Elapsed() << std::endl;
	  for (int i = iu-1; i>= il-1; i--){
	    evals_tmp[i] = evals_tmp[i - (il-1)];
	    if (il>1) evals_tmp[i-(il-1)]=0.;
	    for (int j = 0; j< NN; j++){
	      evec_tmp[i*NN + j] = evec_tmp[(i - (il-1)) * NN + j];
	      if (il>1) evec_tmp[(i-(il-1)) * NN + j]=0.;
	    }
	  }
	}
	{
	  //        QMP_sum_double_array(evals_tmp,NN);
	  //        QMP_sum_double_array((double *)evec_tmp,NN*NN);
	  grid->GlobalSumVector(&evals_tmp[0],NN);
	  grid->GlobalSumVector(&evec_tmp[0],NN*NN);
	}
      } 
      // cheating a bit. It is better to sort instead of just reversing it, but the document of the routine says evals are sorted in increasing order. qr gives evals in decreasing order.
      for(int i=0;i<NN;i++){
	for(int j=0;j<NN;j++)
	  Qt[(NN-1-i)*N2+j]=evec_tmp[i*NN + j];
	lmd [NN-1-i]=evals_tmp[i];
      }

      std::cout << GridLogMessage << "diagonalize_lapack complete\n";
    }
#undef LAPACK_INT 
#endif


    void diagonalize(DenseVector<RealD>& lmd,
		     DenseVector<RealD>& lme, 
		     int N2,
		     int N1,
		     DenseVector<RealD>& Qt,
		     GridBase *grid)
    {

#ifdef USE_LAPACK_IRL
    const int check_lapack=0; // just use lapack if 0, check against lapack if 1

    if(!check_lapack)
	return diagonalize_lapack(lmd,lme,N2,N1,Qt,grid);

	DenseVector <RealD> lmd2(N1);
	DenseVector <RealD> lme2(N1);
	DenseVector<RealD> Qt2(N1*N1);
         for(int k=0; k<N1; ++k){
	    lmd2[k] = lmd[k];
	    lme2[k] = lme[k];
	  }
         for(int k=0; k<N1*N1; ++k)
	Qt2[k] = Qt[k];

//	diagonalize_lapack(lmd2,lme2,Nm2,Nm,Qt,grid);
#endif

      int Niter = 10000*N1;
      int kmin = 1;
      int kmax = N2;
      // (this should be more sophisticated)

      for(int iter=0; ; ++iter){
      if ( (iter+1)%(100*N1)==0) 
      std::cout<<GridLogMessage << "[QL method] Not converged - iteration "<<iter+1<<"\n";

	// determination of 2x2 leading submatrix
	RealD dsub = lmd[kmax-1]-lmd[kmax-2];
	RealD dd = sqrt(dsub*dsub + 4.0*lme[kmax-2]*lme[kmax-2]);
	RealD Dsh = 0.5*(lmd[kmax-2]+lmd[kmax-1] +dd*(dsub/fabs(dsub)));
	// (Dsh: shift)
	
	// transformation
	qr_decomp(lmd,lme,N2,N1,Qt,Dsh,kmin,kmax);
	
	// Convergence criterion (redef of kmin and kamx)
	for(int j=kmax-1; j>= kmin; --j){
	  RealD dds = fabs(lmd[j-1])+fabs(lmd[j]);
	  if(fabs(lme[j-1])+dds > dds){
	    kmax = j+1;
	    goto continued;
	  }
	}
	Niter = iter;
#ifdef USE_LAPACK_IRL
    if(check_lapack){
	const double SMALL=1e-8;
	diagonalize_lapack(lmd2,lme2,N2,N1,Qt2,grid);
	DenseVector <RealD> lmd3(N2);
         for(int k=0; k<N2; ++k) lmd3[k]=lmd[k];
        _sort.push(lmd3,N2);
        _sort.push(lmd2,N2);
         for(int k=0; k<N2; ++k){
	    if (fabs(lmd2[k] - lmd3[k]) >SMALL)  std::cout<<GridLogMessage <<"lmd(qr) lmd(lapack) "<< k << ": " << lmd2[k] <<" "<< lmd3[k] <<std::endl;
//	    if (fabs(lme2[k] - lme[k]) >SMALL)  std::cout<<GridLogMessage <<"lme(qr)-lme(lapack) "<< k << ": " << lme2[k] - lme[k] <<std::endl;
	  }
         for(int k=0; k<N1*N1; ++k){
//	    if (fabs(Qt2[k] - Qt[k]) >SMALL)  std::cout<<GridLogMessage <<"Qt(qr)-Qt(lapack) "<< k << ": " << Qt2[k] - Qt[k] <<std::endl;
	}
    }
#endif
	return;

      continued:
	for(int j=0; j<kmax-1; ++j){
	  RealD dds = fabs(lmd[j])+fabs(lmd[j+1]);
	  if(fabs(lme[j])+dds > dds){
	    kmin = j+1;
	    break;
	  }
	}
      }
      std::cout<<GridLogMessage << "[QL method] Error - Too many iteration: "<<Niter<<"\n";
      abort();
    }

#if 1
    template<typename T>
    static RealD normalise(T& v) 
    {
      RealD nn = norm2(v);
      nn = sqrt(nn);
      v = v * (1.0/nn);
      return nn;
    }

    void orthogonalize(Field& w,
		       BasisFieldVector<Field>& evec,
		       int k)
    {
      double t0=-usecond()/1e6;

      evec.orthogonalize(w,k);

      normalise(w);
      t0+=usecond()/1e6;
      OrthoTime +=t0;
    }

    void setUnit_Qt(int Nm, DenseVector<RealD> &Qt) {
      for(int i=0; i<Qt.size(); ++i) Qt[i] = 0.0;
      for(int k=0; k<Nm; ++k) Qt[k + k*Nm] = 1.0;
    }

/* Rudy Arthur's thesis pp.137
------------------------
Require: M > K P = M − K †
Compute the factorization AVM = VM HM + fM eM 
repeat
  Q=I
  for i = 1,...,P do
    QiRi =HM −θiI Q = QQi
    H M = Q †i H M Q i
  end for
  βK =HM(K+1,K) σK =Q(M,K)
  r=vK+1βK +rσK
  VK =VM(1:M)Q(1:M,1:K)
  HK =HM(1:K,1:K)
  →AVK =VKHK +fKe†K † Extend to an M = K + P step factorization AVM = VMHM + fMeM
until convergence
 */

    void calc(DenseVector<RealD>& eval,
	      BasisFieldVector<Field>& evec,
	      const Field& src,
	      int& Nconv,
	      bool reverse,
	      int SkipTest)
      {

	GridBase *grid = evec._v[0]._grid;//evec.get(0 + evec_offset)._grid;
	assert(grid == src._grid);

	std::cout<<GridLogMessage << " -- Nk = " << Nk << " Np = "<< Np << std::endl;
	std::cout<<GridLogMessage << " -- Nm = " << Nm << std::endl;
	std::cout<<GridLogMessage << " -- size of eval   = " << eval.size() << std::endl;
	std::cout<<GridLogMessage << " -- size of evec  = " << evec.size() << std::endl;
	
	assert(Nm <= evec.size() && Nm <= eval.size());

	// quickly get an idea of the largest eigenvalue to more properly normalize the residuum
	RealD evalMaxApprox = 0.0;
	{
	  auto src_n = src;
	  auto tmp = src;
	  const int _MAX_ITER_IRL_MEVAPP_ = 50;
	  for (int i=0;i<_MAX_ITER_IRL_MEVAPP_;i++) {
	    _HermOpTest(src_n,tmp);
	    RealD vnum = real(innerProduct(src_n,tmp)); // HermOp.
	    RealD vden = norm2(src_n);
	    RealD na = vnum/vden;
	    if (fabs(evalMaxApprox/na - 1.0) < 0.05)
	      i=_MAX_ITER_IRL_MEVAPP_;
	    evalMaxApprox = na;
	    std::cout << GridLogMessage << " Approximation of largest eigenvalue: " << evalMaxApprox << std::endl;
	    src_n = tmp;
	  }
	}
	
	DenseVector<RealD> lme(Nm);  
	DenseVector<RealD> lme2(Nm);
	DenseVector<RealD> eval2(Nm);
	DenseVector<RealD> eval2_copy(Nm);
	DenseVector<RealD> Qt(Nm*Nm);


	Field f(grid);
	Field v(grid);
  
	int k1 = 1;
	int k2 = Nk;

	Nconv = 0;

	RealD beta_k;
  
	// Set initial vector
	evec[0] = src;
	normalise(evec[0]);
	std:: cout<<GridLogMessage <<"norm2(evec[0])= " << norm2(evec[0])<<std::endl;
	
	// Initial Nk steps
	OrthoTime=0.;
	double t0=usecond()/1e6;
	for(int k=0; k<Nk; ++k) step(eval,lme,evec,f,Nm,k);
	double t1=usecond()/1e6;
	std::cout<<GridLogMessage <<"IRL::Initial steps: "<<t1-t0<< "seconds"<<std::endl; t0=t1;
	std::cout<<GridLogMessage <<"IRL::Initial steps:OrthoTime "<<OrthoTime<< "seconds"<<std::endl;
	t1=usecond()/1e6;

	// Restarting loop begins
	for(int iter = 0; iter<Niter; ++iter){
	  
	  std::cout<<GridLogMessage<<"\n Restart iteration = "<< iter << std::endl;
	  
	  // 
	  // Rudy does a sort first which looks very different. Getting fed up with sorting out the algo defs.
	  // We loop over 
	  //
	  OrthoTime=0.;
	  for(int k=Nk; k<Nm; ++k) step(eval,lme,evec,f,Nm,k);
	  t1=usecond()/1e6;
	  std::cout<<GridLogMessage <<"IRL:: "<<Np <<" steps: "<<t1-t0<< "seconds"<<std::endl; t0=t1;
	  std::cout<<GridLogMessage <<"IRL::Initial steps:OrthoTime "<<OrthoTime<< "seconds"<<std::endl;
	  f *= lme[Nm-1];
	  
	  t1=usecond()/1e6;

	  
	  // getting eigenvalues
	  for(int k=0; k<Nm; ++k){
	    eval2[k] = eval[k+k1-1];
	    lme2[k] = lme[k+k1-1];
	  }
	  setUnit_Qt(Nm,Qt);
	  diagonalize(eval2,lme2,Nm,Nm,Qt,grid);
	  t1=usecond()/1e6;
	  std::cout<<GridLogMessage <<"IRL:: diagonalize: "<<t1-t0<< "seconds"<<std::endl; t0=t1;
	  
	  // sorting
	  eval2_copy = eval2;

	  _sort.push(eval2,Nm);
	  t1=usecond()/1e6;
	  std::cout<<GridLogMessage <<"IRL:: eval sorting: "<<t1-t0<< "seconds"<<std::endl; t0=t1;
	  
	  // Implicitly shifted QR transformations
	  setUnit_Qt(Nm,Qt);
	  for(int ip=0; ip<k2; ++ip){
	    std::cout<<GridLogMessage << "eval "<< ip << " "<< eval2[ip] << std::endl;
	  }

	  for(int ip=k2; ip<Nm; ++ip){ 
	    std::cout<<GridLogMessage << "qr_decomp "<< ip << " "<< eval2[ip] << std::endl;
	    qr_decomp(eval,lme,Nm,Nm,Qt,eval2[ip],k1,Nm);
	    
	  }
	  t1=usecond()/1e6;
	  std::cout<<GridLogMessage <<"IRL::qr_decomp: "<<t1-t0<< "seconds"<<std::endl; t0=t1;
	  assert(k2<Nm);
	  

	  assert(k2<Nm);
	  assert(k1>0);
	  evec.rotate(Qt,k1-1,k2+1,0,Nm,Nm);
	  
	  t1=usecond()/1e6;
	  std::cout<<GridLogMessage <<"IRL::QR rotation: "<<t1-t0<< "seconds"<<std::endl; t0=t1;
	  fflush(stdout);
	  
	  // Compressed vector f and beta(k2)
	  f *= Qt[Nm-1+Nm*(k2-1)];
	  f += lme[k2-1] * evec[k2];
	  beta_k = norm2(f);
	  beta_k = sqrt(beta_k);
	  std::cout<<GridLogMessage<<" beta(k) = "<<beta_k<<std::endl;
	  
	  RealD betar = 1.0/beta_k;
	  evec[k2] = betar * f;
	  lme[k2-1] = beta_k;
	  
	  // Convergence test
	  for(int k=0; k<Nm; ++k){    
	    eval2[k] = eval[k];
	    lme2[k] = lme[k];

	    std::cout<<GridLogMessage << "eval2[" << k << "] = " << eval2[k] << std::endl;
	  }
	  setUnit_Qt(Nm,Qt);
	  diagonalize(eval2,lme2,Nk,Nm,Qt,grid);
	  t1=usecond()/1e6;
	  std::cout<<GridLogMessage <<"IRL::diagonalize: "<<t1-t0<< "seconds"<<std::endl; t0=t1;
	  
	  
	  Nconv = 0;
	  
	  if (iter >= Nminres) {
	    std::cout << GridLogMessage << "Rotation to test convergence " << std::endl;
	    
	    Field ev0_orig(grid);
	    ev0_orig = evec[0];
	    
	    evec.rotate(Qt,0,Nk,0,Nk,Nm);
	    
	    {
	      std::cout << GridLogMessage << "Test convergence" << std::endl;
	      Field B(grid);
	      
	      for(int j = 0; j<Nk; j+=SkipTest){
		B=evec[j];
		//std::cout << "Checkerboard: " << evec[j].checkerboard << std::endl; 
		B.checkerboard = evec[0].checkerboard;

		_HermOpTest(B,v);
		
		RealD vnum = real(innerProduct(B,v)); // HermOp.
		RealD vden = norm2(B);
		RealD vv0 = norm2(v);
		eval2[j] = vnum/vden;
		v -= eval2[j]*B;
		RealD vv = norm2(v) / ::pow(evalMaxApprox,2.0);
		std::cout.precision(13);
		std::cout<<GridLogMessage << "[" << std::setw(3)<< std::setiosflags(std::ios_base::right) <<j<<"] "
			 <<"eval = "<<std::setw(25)<< std::setiosflags(std::ios_base::left)<< eval2[j] << " (" << eval2_copy[j] << ")"
			 <<" |H B[i] - eval[i]B[i]|^2 / evalMaxApprox^2 " << std::setw(25)<< std::setiosflags(std::ios_base::right)<< vv
			 <<" "<< vnum/(sqrt(vden)*sqrt(vv0))
			 << " norm(B["<<j<<"])="<< vden <<std::endl;
		
		// change the criteria as evals are supposed to be sorted, all evals smaller(larger) than Nstop should have converged
		if((vv<eresid*eresid) && (j == Nconv) ){
		  Nconv+=SkipTest;
		}
	      }
	      
	      // test if we converged, if so, terminate
	      t1=usecond()/1e6;
	      std::cout<<GridLogMessage <<"IRL::convergence testing: "<<t1-t0<< "seconds"<<std::endl; t0=t1;
	      
	      std::cout<<GridLogMessage<<" #modes converged: "<<Nconv<<std::endl;
	      
	      if( Nconv>=Nstop || beta_k < betastp){
		goto converged;
	      }
	      
	      std::cout << GridLogMessage << "Rotate back" << std::endl;
	      //B[j] +=Qt[k+_Nm*j] * _v[k]._odata[ss];
	      {
		Eigen::MatrixXd qm = Eigen::MatrixXd::Zero(Nk,Nk);
		for (int k=0;k<Nk;k++)
		  for (int j=0;j<Nk;j++)
		    qm(j,k) = Qt[k+Nm*j];
		GridStopWatch timeInv;
		timeInv.Start();
		Eigen::MatrixXd qmI = qm.inverse();
		timeInv.Stop();
		DenseVector<RealD> QtI(Nm*Nm);
		for (int k=0;k<Nk;k++)
		  for (int j=0;j<Nk;j++)
		    QtI[k+Nm*j] = qmI(j,k);
		
		RealD res_check_rotate_inverse = (qm*qmI - Eigen::MatrixXd::Identity(Nk,Nk)).norm(); // sqrt( |X|^2 )
		assert(res_check_rotate_inverse < 1e-7);
		evec.rotate(QtI,0,Nk,0,Nk,Nm);
		
		axpy(ev0_orig,-1.0,evec[0],ev0_orig);
		std::cout << GridLogMessage << "Rotation done (in " << timeInv.Elapsed() << " = " << timeInv.useconds() << " us" <<
		  ", error = " << res_check_rotate_inverse << 
		  "); | evec[0] - evec[0]_orig | = " << ::sqrt(norm2(ev0_orig)) << std::endl;
	      }
	    }
	  } else {
	    std::cout << GridLogMessage << "iter < Nminres: do not yet test for convergence\n";
	  } // end of iter loop
	}

	std::cout<<GridLogMessage<<"\n NOT converged.\n";
	abort();
	
      converged:

	if (SkipTest == 1) {
	  eval = eval2;
	} else {

	  // test quickly
	  for (int j=0;j<Nstop;j+=SkipTest) {
	    std::cout<<GridLogMessage << "Eigenvalue[" << j << "] = " << eval2[j] << " (" << eval2_copy[j] << ")" << std::endl;
	  }

	  eval2_copy.resize(eval2.size());
	  eval = eval2_copy;
	}

	evec.sortInPlace(eval,reverse);

	{
	  
	 // test
	 for (int j=0;j<Nstop;j++) {
	   std::cout<<GridLogMessage << " |e[" << j << "]|^2 = " << norm2(evec[j]) << std::endl;
	 }
       }
       
       //_sort.push(eval,evec,Nconv);
       //evec.sort(eval,Nconv);
       
       std::cout<<GridLogMessage << "\n Converged\n Summary :\n";
       std::cout<<GridLogMessage << " -- Iterations  = "<< Nconv  << "\n";
       std::cout<<GridLogMessage << " -- beta(k)     = "<< beta_k << "\n";
       std::cout<<GridLogMessage << " -- Nconv       = "<< Nconv  << "\n";
      }
#endif

    };

}
#endif

