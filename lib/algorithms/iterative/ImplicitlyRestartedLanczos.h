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
//#include <set>
//#include <list>

#include <string.h> //memset
#ifdef USE_LAPACK
#ifdef USE_MKL
#include<mkl_lapack.h>
#else
void LAPACK_dstegr(char *jobz, char *range, int *n, double *d, double *e,
                   double *vl, double *vu, int *il, int *iu, double *abstol,
                   int *m, double *w, double *z, int *ldz, int *isuppz,
                   double *work, int *lwork, int *iwork, int *liwork,
                   int *info);
//#include <lapacke/lapacke.h>
#endif
#endif
#include "DenseMatrix.h"
#include "EigenSort.h"
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

  int _nd, _blocks, _cf_size, _cf_block_size, _o_blocks, _block_sites;
  
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

};

template<class Field>
class BlockedField {
public:
  typedef typename Field::scalar_type Coeff_t;
  typedef typename FieldHP::scalar_type CoeffHP_t;
  typedef typename Field::vector_type vCoeff_t;
  typedef typename FieldHP::vector_type vCoeffHP_t;

  std::vector< vCoeff_t > _c;
  Field _f;

  BlockedField(GridBase* value) : _f(value) {
  }
};

template<class Field, class FieldHP>
class BlockedFieldVector {
 public:
  int _Nm,  // number of total vectors
    _Nfull; // number of vectors kept in full precision

  typedef typename Field::scalar_type Coeff_t;
  typedef typename FieldHP::scalar_type CoeffHP_t;
  typedef typename Field::vector_type vCoeff_t;
  typedef typename FieldHP::vector_type vCoeffHP_t;
  typedef typename Field::vector_object vobj;
  typedef typename vobj::scalar_object sobj;

  BlockedGrid<FieldHP> _bgrid;
  BlockedGrid<Field> _bgridLP;
  
  std::vector<Field> _v; // _Nfull vectors
  std::vector< vCoeffHP_t > _c; 

  bool _full_locked;
  
  //public:

  BlockedFieldVector(int Nm,GridBase* value,GridBase* valueHP,int Nfull,const std::vector<int>& block_size) : 
  _Nfull(Nfull), _Nm(Nm), _v(Nfull,value), _bgrid(valueHP,block_size), _bgridLP(value,block_size), _full_locked(false) {

    std::cout << GridLogMessage << "BlockedFieldVector initialized:\n";
    std::cout << GridLogMessage << " Nm = " << Nm << "\n";
    std::cout << GridLogMessage << " Nfull = " << Nfull << "\n";
    std::cout << GridLogMessage << " Size of coefficients = " << 
      ((double)_c.size()*sizeof(Coeff_t) / 1024./1024./1024.) << " GB\n";
    std::cout << GridLogMessage << " Size of full vectors = " << 
      ((double)_v.size()*sizeof(Coeff_t)*_bgrid._cf_size / 1024./1024./1024.) << " GB\n";
  }
  
  ~BlockedFieldVector() {
  
  }

#define cidx(i,b,j) ((int64_t)b + (int64_t)_bgrid._o_blocks * (int64_t)(j + _Nfull*i))
  
  void lock_in_full_vectors() {

    assert(!_full_locked);
    _full_locked = true;

    // resize appropriately
    _c.resize(_Nm*_Nfull*_bgrid._o_blocks);
#pragma omp parallel for
    for (int64_t i=0;i<_c.size();i++)
      _c[i] = 0.0;

    GridStopWatch sw;
    sw.Start();
    // orthogonalize local blocks and create coefficients for first Nfull vectors
    for (int i=0;i<_Nfull;i++) {
      
      FieldHP vi(_bgrid._grid);
      precisionChange(vi,_v[i]);
      
      // |i> -= <j|i> |j>
      for (int j=0;j<i;j++) {

	FieldHP vj(_bgrid._grid);
	precisionChange(vj,_v[j]);

#pragma omp parallel for
	for (int b=0;b<_bgrid._o_blocks;b++) {
	  vCoeffHP_t v = _bgrid.block_sp(b,vj,vi);
	  _c[ cidx(i,b,j) ] = v;
	  _bgrid.block_caxpy(b,vi,-v,vj,vi);
	}
      }
      
#pragma omp parallel for
      for (int b=0;b<_bgrid._o_blocks;b++) {
	vCoeffHP_t nrm = _bgrid.block_sp(b,vi,vi);
	_c[ cidx(i,b,i) ] = sqrt(nrm);
	_bgrid.block_cscale(b,1.0 / sqrt(nrm),vi);
      }
      
      precisionChange(_v[i],vi); // copy back
    }
    sw.Stop();
    std::cout << GridLogMessage << "Gram-Schmidt to create blocked basis took " << sw.Elapsed() << std::endl;

  }

  BlockedField<FieldHP> get_blocked(int i) {

    BlockedField<FieldHP> ret(_bgrid._grid);
    ret = zero;
    ret.checkerboard = _v[0].checkerboard;
    
    for (int j=0;j<_Nfull;j++) {
      FieldHP vj(_bgrid._grid);
      precisionChange(vj,_v[j]);
#pragma omp parallel for
      for (int b=0;b<_bgrid._o_blocks;b++)
	_bgrid.block_caxpy(b,ret,_c[ cidx(i,b,j) ],vj,ret);
    }
    
    return ret;
  }

  void put_blocked(int i, const FieldHP& rhs) {

    FieldHP tmp(_bgrid._grid);
    tmp = rhs;

    for (int j=0;j<_Nfull;j++) {
      FieldHP vj(_bgrid._grid);
      precisionChange(vj,_v[j]);
      
#pragma omp parallel for
      for (int b=0;b<_bgrid._o_blocks;b++) {
	// |rhs> -= <j|rhs> |j>
	_c[ cidx(i,b,j) ] = _bgrid.block_sp(b,vj,tmp);
      }
    }

  }

  FieldHP get(int i) {
    if (!_full_locked) {

      assert(i < _Nfull);
      FieldHP tmp(_bgrid._grid);
      precisionChange(tmp,_v[i]);
      return tmp;

    } else {

      return get_blocked(i);

    }
  }

  void put(int i, const FieldHP& v) {
    //std::cout << GridLogMessage << "::put(" << i << ")\n";

    if (!_full_locked && i >= _Nfull) {
      // lock in smallest vectors so we can build on them
      //Field test = _v[_Nfull - 1];
      lock_in_full_vectors();
      //axpy(test,-1.0,get_blocked(_Nfull - 1),test);
      //std::cout << GridLogMessage << "Error of lock_in_full_vectors: " << norm2(test) << std::endl;
    }

    if (!_full_locked) {
      assert(i < _Nfull);
      //_v[i] = v;
      precisionChange(_v[i],v);
    } else {
      put_blocked(i,v);

#if 0 // this is wasted time now that we have tested the code for correctness
      FieldHP test = get_blocked(i);
      RealD nrm2b = norm2(test);
      axpy(test,-1.0,v,test);
      std::cout << GridLogMessage << "Error of vector: " << norm2(test) << " nrm2 = " << norm2(v) << " vs " << nrm2b << std::endl;
#endif
    }
  }

  void deflate(const std::vector<RealD>& eval,const std::vector<int>& idx,int N,const Field& src_orig,Field& result) {
    result = zero;
    Field tmp(result);
    for (int i=0;i<N;i++) {
      int j = idx[i];
      precisionChange(tmp,get(j));
      axpy(result,TensorRemove(innerProduct(tmp,src_orig)) / eval[j],tmp,result);
    }
  }

  void orthogonalize(FieldHP& whp, int k, int evec_offset) {

    if (!_full_locked) {

      //#define LANC_ORTH_HIGH_PRECISION
#ifdef LANC_ORTH_HIGH_PRECISION
      for(int j=0; j<k; ++j){
	FieldHP evec_j(_bgrid._grid);
	precisionChange(evec_j,_v[j + evec_offset]);
	CoeffHP_t ip = innerProduct(evec_j,whp);
	whp = whp - ip*evec_j;
	//Field evec_j = _v[j + evec_offset];
	//Coeff_t ip = (Coeff_t)innerProduct(evec_j,w); // are the evecs normalised? ; this assumes so.
	//w = w - ip * evec_j;
      }
#else
      Field w(_v[0]._grid);
      precisionChange(w,whp);
      for(int j=0; j<k; ++j){
	Coeff_t ip = (Coeff_t)innerProduct(_v[j + evec_offset],w);
	w = w - ip*_v[j + evec_offset];
      }
      precisionChange(whp,w);
      
#endif

    } else {

      // first represent w in blocks
      std::vector< vCoeffHP_t > cw;
      cw.resize(_Nfull*_bgrid._o_blocks);

      for (int j=0;j<_Nfull;j++) {
	FieldHP vj(_bgrid._grid);
	precisionChange(vj,_v[j]);

#pragma omp parallel for
	for (int b=0;b<_bgrid._o_blocks;b++) {
	  cw[cidx(0,b,j)] = _bgrid.block_sp(b,vj,whp);
	}
      }

      // now can ortho just in coefficient space, should be much faster
      // w     = cw[ cidx(0,b,j) ] * _v[j,b]
      // _v[i] = _c[ cidx(i,b,j) ] * _v[j,b]

      CoeffHP_t ip;
#pragma omp parallel shared(ip)
      {
	for(int j=0; j<k; ++j) {

#pragma omp barrier
#pragma omp single
	  {
	    ip = 0;
	  }

	  //Field evec_j = _v[j + evec_offset];
	  // evec_j = _c[ cidx(j + evec_offset,b,l) ] * _v[l,b]
	  
	  // ip = innerProduct(evec_j,w);
	  // ip = conj(_c[ cidx(j + evec_offset,b,l) ]) * innerProduct(_v[l,b],_v[l',b']) * cw[ cidx(0,b',l') ]
	  //    = conj(_c[ cidx(j + evec_offset,b,l) ]) * cw[ cidx(0,b,l) ]
	  CoeffHP_t ipl = 0;
#pragma omp for
	  for (int b=0;b<_bgrid._o_blocks;b++) {
	    for (int l=0;l<_Nfull;l++)
	      ipl += Reduce(conjugate(_c[ cidx(j + evec_offset,b,l) ]) * cw[ cidx(0,b,l) ]);
	  }

#pragma omp critical
	  {
	    ip += ipl;
	  }

#pragma omp barrier
#pragma omp single
	  {
	    _bgrid._grid->GlobalSum(ip);
	    //std::cout << GridLogMessage << "Overlap of " << k << " with " << j << " is " << ipl << std::endl;
	  }


	  //w = w - ip * evec_j;
#pragma omp for
	  for (int b=0;b<_bgrid._o_blocks;b++) {
	    for (int l=0;l<_Nfull;l++)
	      cw[ cidx(0,b,l) ] -= ip* _c[ cidx(j+evec_offset,b,l) ];
	  }
	}
      }

      // reconstruct w
      whp = zero;
      for (int j=0;j<_Nfull;j++) {
	FieldHP vj(_bgrid._grid);
	precisionChange(vj,_v[j]);

#pragma omp parallel for
	for (int b=0;b<_bgrid._o_blocks;b++) {
	  _bgrid.block_caxpy(b,whp,cw[ cidx(0,b,j) ],vj,whp);
	}
      }

    }

  }

  void rotate(DenseVector<RealD>& Qt,int j0, int j1, int k0,int k1,int Nm,int evec_offset) {
   
    if (!_full_locked) {

      GridBase* grid = _v[0]._grid;
      
#pragma omp parallel
      {
	std::vector < vobj > B(Nm);
	
#pragma omp for
	for(int ss=0;ss < grid->oSites();ss++){
	  for(int j=j0; j<j1; ++j) B[j]=0.;
	  
	  for(int j=j0; j<j1; ++j){
	    for(int k=k0; k<k1; ++k){
	      B[j] +=Qt[k+Nm*j] * _v[k + evec_offset]._odata[ss];
	    }
	  }
	  for(int j=j0; j<j1; ++j){
	    _v[j + evec_offset]._odata[ss] = B[j];
	  }
	}
      }
    } else {

      // B_j = Q_jk A_k
      // A_k = _c[ k, bj ] _v[bj]
      // B_j = Q_jk _c[ k, bl ] _v[bl]
      // -> _c[ j, bl ] = Q_jk _c[ k, bl ]
#pragma omp parallel
      {
        std::vector<vCoeffHP_t> c0;
	c0.resize(_Nfull*Nm);

#pragma omp for
        for (int b=0;b<_bgrid._o_blocks;b++) {
          for (int l=0;l<_Nfull;l++) {
	    for(int j=j0; j<j1; ++j){
	      vCoeffHP_t& cc = c0[l + _Nfull*j];
	      cc = 0.0;
	      for(int k=k0; k<k1; ++k){
		cc +=Qt[k+Nm*j] * _c[ cidx((k+evec_offset),b,l) ];
	      }
	    }
          }
	  for (int l=0;l<_Nfull;l++) {
	    for(int j=j0; j<j1; ++j){
	      _c[ cidx((j+evec_offset),b,l) ] = c0[l + _Nfull*j];
	    }
	  }
	}
      }
      
    }
  }

  size_t size() const {
    return _Nm;
  }

  std::vector<int> getIndex(DenseVector<RealD>& sort_vals) {

    std::vector<int> idx(sort_vals.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
	 [&sort_vals](int i1, int i2) {return sort_vals[i1] < sort_vals[i2];});

    return idx;
  }


  // zlib's crc32 gets 0.4 GB/s on KNL single thread
  // below gets 4.8 GB/s
  uint32_t crc32_threaded(unsigned char* data, int64_t len, uint32_t previousCrc32 = 0) {

    return crc32(previousCrc32,data,len);

    // below needs further tuning/testing
    std::vector<uint32_t> partials;
    std::vector<int64_t> lens;
    int64_t len_part;

#pragma omp parallel shared(partials,lens,len_part)
    {
      int threads = omp_get_num_threads();
      int thread  = omp_get_thread_num();

#pragma omp single
      {
	partials.resize(threads);
	lens.resize(threads);
	assert(len % threads == 0); // for now 64 divides all our data, easy to generalize
	len_part = len / threads;
      }

#pragma omp barrier

      partials[thread] = crc32(thread == 0 ? previousCrc32 : 0x0,data + len_part * thread,len_part);
      lens[thread] = len_part;

      // reduction
      while (lens.size() > 1) {

	uint32_t com_val;
	int64_t com_len;
	if (thread % 2 == 0) {
	  if (thread + 1 < (int)partials.size()) {
	    com_val = crc32_combine(partials[thread],partials[thread+1],lens[thread+1]);
	    com_len = lens[thread] + lens[thread+1];
	  } else if (thread + 1 == (int)partials.size()) {
	    com_val = partials[thread];
	    com_len = lens[thread];
	  } else {
	    com_len = -1; // inactive thread
	  }
	} else {
	  com_len = -1;
	}

	//std::cout << "Reducing in thread " << thread << " from lens.size() = " << lens.size() << " found " << com_len << std::endl;
	
#pragma omp barrier

#pragma omp single
	{
	  partials.resize(0);
	  lens.resize(0);
	}

#pragma omp barrier
	
#pragma omp critical
	{
	  if (com_len != -1) {
	    partials.push_back(com_val);
	    lens.push_back(com_len);
	  }
	}

#pragma omp barrier
      }	

    }

    return partials[0];
  }

  int get_bfm_index( int* pos, int co, int* s ) {
    
    int ls = s[0];
    int NtHalf = s[4] / 2;
    int simd_coor = pos[4] / NtHalf;
    int regu_coor = (pos[1] + s[1] * (pos[2] + s[2] * ( pos[3] + s[3] * (pos[4] % NtHalf) ) )) / 2;
     
    return regu_coor * ls * 48 + pos[0] * 48 + co * 4 + simd_coor * 2;
  }

  bool read_argonne(const char* dir, const std::vector<int>& cnodes) {

    assert(!_full_locked);

    // this is slow code to read the argonne file format for debugging purposes
    std::vector<int> nodes = cnodes;
    std::vector<int> slot_lvol, lvol;
    int _nd = (int)nodes.size();
    int i, ntotal = 1;
    int64_t lsites = 1;
    int64_t slot_lsites = 1;
    for (i=0;i<_nd;i++) {
      slot_lvol.push_back(_bgrid._grid->FullDimensions()[i] / nodes[i]);
      lvol.push_back(_bgrid._grid->FullDimensions()[i] / _bgrid._grid->_processors[i]);
      lsites *= lvol.back();
      slot_lsites *= slot_lvol.back();
      ntotal *= nodes[i];
    }

    int nperdir = ntotal / 32;
    if (nperdir < 1)
      nperdir=1;
    std::cout << GridLogMessage << " Read " << dir << " nodes = " << nodes << std::endl;
    std::cout << GridLogMessage << " lvol = " << lvol << std::endl;

    std::map<int, std::vector<int> > slots;


    {
      std::vector<int> lcoor, gcoor, scoor;
      lcoor.resize(_nd); gcoor.resize(_nd);  scoor.resize(_nd);
      
      // first create mapping of indices to slots
      for (int lidx = 0; lidx < lsites; lidx++) {
	Lexicographic::CoorFromIndex(lcoor,lidx,lvol);
	for (int i=0;i<_nd;i++) {
	  gcoor[i] = lcoor[i] + _bgrid._grid->_processor_coor[i]*lvol[i];
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

    // now load one slot at a time and fill the vector
    for (auto sl=slots.begin();sl!=slots.end();sl++) {
      std::vector<int>& idx = sl->second;
      int slot = sl->first;
      std::vector<float> rdata;

      char buf[4096];

      sprintf(buf,"%s/checksums.txt",dir);
      FILE* f = fopen(buf,"rt");
      if (!f)
	return false;

      for (int l=0;l<3+slot;l++)
	fgets(buf,sizeof(buf),f);
      uint32_t crc_exp = strtol(buf, NULL, 16);
      fclose(f);

      // load one slot vector
      sprintf(buf,"%s/%2.2d/%10.10d",dir,slot/nperdir,slot);
      f = fopen(buf,"rb");
      if (!f)
	return false;
      
      fseeko(f,0,SEEK_END);
      off_t total_size = ftello(f);
      fseeko(f,0,SEEK_SET);

      int64_t size = slot_lsites / 2 * 24 * 4;
      rdata.resize(size);

      assert(total_size % size == 0);

      _Nfull = total_size / size;
      _v.resize(_Nfull,_v[0]);
      
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
	_v[nev].checkerboard = Odd;
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
	      gcoor[i] = lcoor[i] + _bgrid._grid->_processor_coor[i]*lvol[i];
	      scoor[i] = gcoor[i] / slot_lvol[i];
	      slcoor[i] = gcoor[i] - scoor[i]*slot_lvol[i];
	    }
	    
	    if ((lcoor[1]+lcoor[2]+lcoor[3]+lcoor[4]) % 2 == 1) {
	      // poke 
	      sobj sc;
	      for (int s=0;s<4;s++)
		for (int c=0;c<3;c++)
		  sc()(s)(c) = *(std::complex<float>*)&rdata[get_bfm_index(&slcoor[0],c+s*3, &slot_lvol[0] )];
	      
	      pokeLocalSite(sc,_v[nev],lcoor);
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

    _bgrid._grid->Barrier();
    std::cout << GridLogMessage  << "Loading complete" << std::endl;

    return true;
  }

  void write_argonne(const char* dir) {

    char buf[4096];

    assert(!_full_locked);

    if (_bgrid._grid->IsBoss()) {
      mkdir(dir,ACCESSPERMS);
      
      for (int i=0;i<32;i++) {
	sprintf(buf,"%s/%2.2d",dir,i);
	mkdir(buf,ACCESSPERMS);
      }
    }

    _bgrid._grid->Barrier(); // make sure directories are ready


    int nperdir = _bgrid._grid->_Nprocessors / 32;
    if (nperdir < 1)
      nperdir=1;
    std::cout << GridLogMessage << " Write " << dir << " nodes = " << _bgrid._grid->_Nprocessors << std::endl;

    int slot;
    Lexicographic::IndexFromCoor(_bgrid._grid->_processor_coor,slot,_bgrid._grid->_processors);
    //printf("Slot: %d <> %d\n",slot, _bgrid._grid->ThisRank());

    sprintf(buf,"%s/%2.2d/%10.10d",dir,slot/nperdir,slot);
    FILE* f = fopen(buf,"wb");
    assert(f);

    int N = (int)_v.size();
    uint32_t crc = 0x0;
    int64_t cf_size = _bgrid._cf_size;
    std::vector< float > rdata(cf_size*2);

    GridStopWatch gsw1,gsw2;

    for (int i=0;i<N;i++) {
      // create buffer and put data in argonne format in there
      std::vector<int> coor(_bgrid._l.size());
      for (coor[1] = 0;coor[1]<_bgrid._l[1];coor[1]++) {
	for (coor[2] = 0;coor[2]<_bgrid._l[2];coor[2]++) {
	  for (coor[3] = 0;coor[3]<_bgrid._l[3];coor[3]++) {
	    for (coor[4] = 0;coor[4]<_bgrid._l[4];coor[4]++) {
	      for (coor[0] = 0;coor[0]<_bgrid._l[0];coor[0]++) {

		if ((coor[1]+coor[2]+coor[3]+coor[4]) % 2 == 1) {
		  // peek
		  sobj sc;
		  peekLocalSite(sc,_v[i],coor);
		  for (int s=0;s<4;s++)
		    for (int c=0;c<3;c++)
		      *(std::complex<float>*)&rdata[get_bfm_index(&coor[0],c+s*3, &_bgrid._l[0] )] = sc()(s)(c);
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
    std::vector<uint32_t> crcs(_bgrid._grid->_Nprocessors);
    for (int i=0;i<_bgrid._grid->_Nprocessors;i++) {
      crcs[i] = 0x0;
    }
    crcs[slot] = crc;
    for (int i=0;i<_bgrid._grid->_Nprocessors;i++) {
      _bgrid._grid->GlobalSum(crcs[i]);
    }

    if (_bgrid._grid->IsBoss()) {
      sprintf(buf,"%s/checksums.txt",dir);
      FILE* f = fopen(buf,"wt");
      assert(f);
      fprintf(f,"00000000\n\n");
      for (int i =0;i<_bgrid._grid->_Nprocessors;i++)
	fprintf(f,"%X\n",crcs[i]);
      fclose(f);

      sprintf(buf,"%s/nodes.txt",dir);
      f = fopen(buf,"wt");
      assert(f);
      for (int i =0;i<(int)_bgrid._grid->_processors.size();i++)
	fprintf(f,"%d\n",_bgrid._grid->_processors[i]);
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

    _bgrid._grid->Barrier();
    std::cout << GridLogMessage  << "Writing complete" << std::endl;

  }

};


template<typename Field>
class IdentityProjector : public LinearFunction<Field> {
public:
  IdentityProjector() {
  }

  void operator()(const Field& in, Field& out) {
    out = in;
  }
};

 template<typename Field,typename FieldHP>
class HighModeProjector : public LinearFunction<Field> {
public:
  int _N;
  BlockedFieldVector<Field,FieldHP>& _evec;

 HighModeProjector(BlockedFieldVector<Field,FieldHP>& evec, int N) : _N(N), _evec(evec) {
  }

  void operator()(const Field& in, Field& out) {
    out = in;
    for (int i = 0;i<_N;i++) {
      Field v = _evec.get(i);
      // |v><v|out>
      axpy(out,-innerProduct(v,out),v,out);
    }
  }
};

 template<typename Field,typename FieldHP>
class HighModeAndBlockProjector : public LinearFunction<Field> {
public:
   BlockedFieldVector<Field,FieldHP>& _evec;

 HighModeAndBlockProjector(BlockedFieldVector<Field,FieldHP>& evec) : _evec(evec) {
  }

  void operator()(const Field& in, Field& out) {
    Field tmp(in);
    tmp = in;

    GridStopWatch gsw1,gsw2;
#if 1 // first keep low modes and try to re-find them
    // (1 - sum_n |n><n|) 
    // first get low modes out
    for (int i = 0;i<_evec._Nfull;i++) {
      Field v = _evec.get(i);
      gsw1.Start();
      axpy(tmp,-innerProduct(v,tmp),v,tmp);
      gsw1.Stop();
    }
#endif

    out = zero;
    out.checkerboard = tmp.checkerboard;
    gsw2.Start();
#pragma omp parallel for
    for (int b=0;b<_evec._bgrid._o_blocks;b++) {
      for (int j=0;j<_evec._Nfull;j++) {
	auto v = _evec._bgrid.block_sp(b,_evec._v[j],tmp);
	_evec._bgrid.block_caxpy(b,out,v,_evec._v[j],out);
      }
    }
    gsw2.Stop();

    /*
      Count flops

      6 per complex multiply, 2 per complex add

      innerProduct:    COUNT_FLOPS_BYTES(8 / 2 * f_size_block, 2*f_size_block*sizeof(OPT));
      axpy:            COUNT_FLOPS_BYTES(8 / 2 * f_size_block, 3*f_size_block*sizeof(OPT));

    */
    double Gflops = _evec._bgrid._cf_size * 2 * (8 / 2) * 2 / 1024./1024./1024. * _evec._Nfull;
    double Gbytes = _evec._bgrid._cf_size * 2 * sizeof(_evec._c[0])/2 * 5 / 1024./1024./1024. * _evec._Nfull;
    double gsw1_s = gsw1.useconds() / 1000. / 1000.;
    double gsw2_s = gsw2.useconds() / 1000. / 1000.;

    std::cout << GridLogMessage << "HighModeProjector norm2 = " << norm2(in) << " -> " << norm2(out) << 
      " at " << Gflops/gsw1_s << " Gflops/s, " << Gflops/gsw2_s << " Gflops/s, " <<
      Gbytes/gsw1_s << " Gbytes/s, " << Gbytes/gsw2_s << " Gbytes/s" <<
      std::endl;
  }
};


template<typename Field,typename FieldHP>
class BlockProjector : public LinearFunction<FieldHP> {
public:
  BlockedFieldVector<Field,FieldHP>& _evec;

  BlockProjector(BlockedFieldVector<Field,FieldHP>& evec) : _evec(evec) {
  }

  void operator()(const FieldHP& in, FieldHP& outhp) {

#if 0
    FieldHP tmp(_evec._bgrid._grid);
    tmp = in;

    outhp = zero;

    outhp.checkerboard = in.checkerboard;
    tmp.checkerboard = in.checkerboard;

    for (int j=0;j<_evec._Nfull;j++) {
      FieldHP vj(_evec._bgrid._grid);
      precisionChange(vj,_evec._v[j]);

#pragma omp parallel for
      for (int b=0;b<_evec._bgrid._o_blocks;b++) {
	_evec._bgrid.block_caxpy(b,outhp,_evec._bgrid.block_sp(b,vj,tmp),vj,outhp);
      }

    }
#else
    Field tmp(_evec._bgridLP._grid);
    precisionChange(tmp,in);

    Field out(_evec._bgridLP._grid);
    out = zero;

    out.checkerboard = in.checkerboard;
    tmp.checkerboard = in.checkerboard;

#pragma omp parallel for
    for (int b=0;b<_evec._bgridLP._o_blocks;b++) {
      for (int j=0;j<_evec._Nfull;j++) {
	auto v = _evec._bgridLP.block_sp(b,_evec._v[j],tmp);
	_evec._bgridLP.block_caxpy(b,out,v,_evec._v[j],out);
	_evec._bgridLP.block_caxpy(b,tmp,-v,_evec._v[j],tmp);
      }
    }

    precisionChange(outhp,out);

#endif
  }
};


 template<class Field>
   class ProjectedSchurOperator :  public SchurOperatorBase<Field> {
 protected:
   SchurOperatorBase<Field> &_Mat;
   LinearFunction<Field> &_Pr;
 public:
   ProjectedSchurOperator (SchurOperatorBase<Field>& Mat, LinearFunction<Field>& Pr): _Mat(Mat), _Pr(Pr) {};
   
   virtual  RealD Mpc      (const Field &in, Field &out) {
     assert(0);
   }
   virtual  RealD MpcDag   (const Field &in, Field &out){
     assert(0);
   }
   virtual void MpcDagMpc(const Field &in, Field &out,RealD &ni,RealD &no) {
     Field tmp(in._grid);
     _Pr(in,out);
     _Mat.MpcDagMpc(out,tmp,ni,no);
     _Pr(tmp,out);
     ni = 0; // OK for current purpose
     no = 0;
   }
 };


/////////////////////////////////////////////////////////////
// Implicitly restarted lanczos
/////////////////////////////////////////////////////////////

 template<class Field,class FieldHP> 
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

    RealD eresid;

    SortEigen<Field> _sort;

    LinearOperatorBase<Field> &_Linop;

    OperatorFunction<Field>   &_poly;

    LinearFunction<FieldHP> &_proj;
    /////////////////////////
    // Constructor
    /////////////////////////
    void init(void){};
    void Abort(int ff, DenseVector<RealD> &evals,  DenseVector<DenseVector<RealD> > &evecs);

    ImplicitlyRestartedLanczos(
			       LinearOperatorBase<Field> &Linop, // op
			       OperatorFunction<Field> & poly,   // polynmial
			       LinearFunction<FieldHP> & proj,
			       int _Nstop, // sought vecs
			       int _Nk, // sought vecs
			       int _Nm, // spare vecs
			       RealD _eresid, // resid in lmdue deficit 
			       int _Niter, // Max iterations
			       int _Nminres) :
      _Linop(Linop),
      _poly(poly),
      _proj(proj),
      Nstop(_Nstop),
      Nk(_Nk),
      Nm(_Nm),
      eresid(_eresid),
      Niter(_Niter),
      Nminres(_Nminres)
    { 
      Np = Nm-Nk; assert(Np>0);
    };

    ImplicitlyRestartedLanczos(
				LinearOperatorBase<Field> &Linop, // op
			       OperatorFunction<Field> & poly,   // polynmial
			       LinearFunction<FieldHP> & proj,
			       int _Nk, // sought vecs
			       int _Nm, // spare vecs
			       RealD _eresid, // resid in lmdue deficit 
			       int _Niter, // Max iterations
			       int _Nminres) : 
      _Linop(Linop),
      _poly(poly),
      _proj(proj),
      Nstop(_Nk),
      Nk(_Nk),
      Nm(_Nm),
      eresid(_eresid),
      Niter(_Niter),
      Nminres(_Nminres)
    { 
      Np = Nm-Nk; assert(Np>0);
    };

    /////////////////////////
    // Sanity checked this routine (step) against Saad.
    /////////////////////////
      void RitzMatrix(BlockedFieldVector<Field,FieldHP>& evec,int k){

#if 0
      if(1) return;

      GridBase *grid = evec[0]._grid;
      Field w(grid);
      std::cout<<GridLogMessage << "RitzMatrix "<<std::endl;
      for(int i=0;i<k;i++){
	_poly(_Linop,evec[i],w);
	std::cout<<GridLogMessage << "["<<i<<"] ";
	for(int j=0;j<k;j++){
	  ComplexD in = innerProduct(evec[j],w);
	  if ( fabs((double)i-j)>1 ) { 
	    if (abs(in) >1.0e-9 )  { 
	      std::cout<<GridLogMessage<<"oops"<<std::endl;
	      abort();
	    } else 
	      std::cout<<GridLogMessage << " 0 ";
	  } else { 
	    std::cout<<GridLogMessage << " "<<in<<" ";
	  }
	}
	std::cout<<GridLogMessage << std::endl;
      }
#endif
    }

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
	      BlockedFieldVector<Field,FieldHP>& evec,
	      FieldHP& w,int Nm,int k,int evec_offset)
    {
      assert( k< Nm );

      GridStopWatch gsw_g,gsw_p,gsw_pr,gsw_cheb,gsw_o;

      gsw_g.Start();
      FieldHP evec_k = evec.get(k + evec_offset);
      gsw_g.Stop();

      Field wLP(evec._v[0]._grid);
      FieldHP tmp(evec_k);
      Field tmpLP(wLP);

      gsw_pr.Start();
      _proj(evec_k,w);
      gsw_pr.Stop();
      gsw_cheb.Start();
      precisionChange(wLP,w);
      _poly(_Linop,wLP,tmpLP);      // 3. wk:=Avk−βkv_{k−1}
      precisionChange(tmp,tmpLP);
      gsw_cheb.Stop();
      gsw_pr.Start();
      _proj(tmp,w);
      gsw_pr.Stop();

#if 0
      // Should we adopt the compression?
      if (k < Nm - 1) {
	evec.put(k+1 + evec_offset,w);
	w = evec.get(k+1+ evec_offset);
      }
#endif

      if(k>0){
	w -= lme[k-1] * evec.get(k-1 + evec_offset);
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
	orthogonalize(w,evec,k,evec_offset); // orthonormalise
      }
      gsw_o.Stop();

      gsw_p.Start();
      if(k < Nm-1) { 
	evec.put(k+1 + evec_offset, w);
	//w = evec.get(k+1 + evec_offset);  // adopt compression for w?
      }
      gsw_p.Stop();

      std::cout << GridLogMessage << "Timing: get=" << gsw_g.Elapsed() <<
	" put="<< gsw_p.Elapsed() <<
	" proj=" << gsw_pr.Elapsed() <<
	" cheb=" << gsw_cheb.Elapsed() <<
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

#ifdef USE_LAPACK
#define LAPACK_INT long long
    void diagonalize_lapack(DenseVector<RealD>& lmd,
		     DenseVector<RealD>& lme, 
		     int N1,
		     int N2,
		     DenseVector<RealD>& Qt,
		     GridBase *grid){
  const int size = Nm;
//  tevals.resize(size);
//  tevecs.resize(size);
  LAPACK_INT NN = N1;
  double evals_tmp[NN];
  double evec_tmp[NN][NN];
  memset(evec_tmp[0],0,sizeof(double)*NN*NN);
//  double AA[NN][NN];
  double DD[NN];
  double EE[NN];
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
  LAPACK_INT iwork[liwork];
  double work[lwork];
  LAPACK_INT isuppz[2*NN];
  char jobz = 'V'; // calculate evals & evecs
  char range = 'I'; // calculate all evals
  //    char range = 'A'; // calculate all evals
  char uplo = 'U'; // refer to upper half of original matrix
  char compz = 'I'; // Compute eigenvectors of tridiagonal matrix
  int ifail[NN];
  long long info;
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
      memset(evals_tmp,0,sizeof(double)*NN);
      if ( il <= NN){
        printf("total=%d node=%d il=%d iu=%d\n",total,node,il,iu);
#ifdef USE_MKL
        dstegr(&jobz, &range, &NN,
#else
        LAPACK_dstegr(&jobz, &range, &NN,
#endif
            (double*)DD, (double*)EE,
            &vl, &vu, &il, &iu, // these four are ignored if second parameteris 'A'
            &tol, // tolerance
            &evals_found, evals_tmp, (double*)evec_tmp, &NN,
            isuppz,
            work, &lwork, iwork, &liwork,
            &info);
        for (int i = iu-1; i>= il-1; i--){
          printf("node=%d evals_found=%d evals_tmp[%d] = %g\n",node,evals_found, i - (il-1),evals_tmp[i - (il-1)]);
          evals_tmp[i] = evals_tmp[i - (il-1)];
          if (il>1) evals_tmp[i-(il-1)]=0.;
          for (int j = 0; j< NN; j++){
            evec_tmp[i][j] = evec_tmp[i - (il-1)][j];
            if (il>1) evec_tmp[i-(il-1)][j]=0.;
          }
        }
      }
      {
//        QMP_sum_double_array(evals_tmp,NN);
//        QMP_sum_double_array((double *)evec_tmp,NN*NN);
         grid->GlobalSumVector(evals_tmp,NN);
         grid->GlobalSumVector((double*)evec_tmp,NN*NN);
      }
    } 
// cheating a bit. It is better to sort instead of just reversing it, but the document of the routine says evals are sorted in increasing order. qr gives evals in decreasing order.
  for(int i=0;i<NN;i++){
    for(int j=0;j<NN;j++)
      Qt[(NN-1-i)*N2+j]=evec_tmp[i][j];
      lmd [NN-1-i]=evals_tmp[i];
  }
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

#ifdef USE_LAPACK
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
#ifdef USE_LAPACK
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

    void orthogonalize(FieldHP& w,
		       BlockedFieldVector<Field,FieldHP>& evec,
		       int k, int evec_offset)
    {
      double t0=-usecond()/1e6;
      typedef typename Field::scalar_type MyComplex;
      MyComplex ip;

      evec.orthogonalize(w,k,evec_offset);

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
	      BlockedFieldVector<Field,FieldHP>& evec,
	      const FieldHP& src,
	      int& Nconv,
	      int evec_offset = 0,
	      bool test_conv_poly = false)
      {

	GridBase *grid = evec._bgrid._grid;//evec.get(0 + evec_offset)._grid;
	assert(grid == src._grid);

	std::cout<<GridLogMessage << " -- Nk = " << Nk << " Np = "<< Np << std::endl;
	std::cout<<GridLogMessage << " -- Nm = " << Nm << std::endl;
	std::cout<<GridLogMessage << " -- evec_offset = " << evec_offset << std::endl;
	std::cout<<GridLogMessage << " -- size of eval   = " << eval.size() << std::endl;
	std::cout<<GridLogMessage << " -- size of evec  = " << evec.size() << std::endl;
	
	assert(Nm <= evec.size() && Nm <= eval.size());
	
	DenseVector<RealD> lme(Nm);  
	DenseVector<RealD> lme2(Nm);
	DenseVector<RealD> eval2(Nm);
	DenseVector<RealD> Qt(Nm*Nm);
	DenseVector<int>   Iconv(Nm);


	FieldHP f(grid);
	FieldHP v(grid);
  
	int k1 = 1;
	int k2 = Nk;

	Nconv = 0;

	RealD beta_k;
  
	// Set initial vector
	// (uniform vector) Why not src??
	//	evec[0] = 1.0;
	FieldHP src_n=src;
	normalise(src_n);
	evec.put(0 + evec_offset,src_n);
	std:: cout<<GridLogMessage <<"norm2(src)= " << norm2(src)<<std::endl;
// << src._grid  << std::endl;
	std:: cout<<GridLogMessage <<"norm2(evec[0])= " << norm2(evec.get(0 + evec_offset)) <<std::endl;
// << evec[0]._grid << std::endl;
	
	// Initial Nk steps
	OrthoTime=0.;
	double t0=usecond()/1e6;
	for(int k=0; k<Nk; ++k) step(eval,lme,evec,f,Nm,k,evec_offset);
	double t1=usecond()/1e6;
	std::cout<<GridLogMessage <<"IRL::Initial steps: "<<t1-t0<< "seconds"<<std::endl; t0=t1;
	std::cout<<GridLogMessage <<"IRL::Initial steps:OrthoTime "<<OrthoTime<< "seconds"<<std::endl;
//	std:: cout<<GridLogMessage <<"norm2(evec[1])= " << norm2(evec[1]) << std::endl;
//	std:: cout<<GridLogMessage <<"norm2(evec[2])= " << norm2(evec[2]) << std::endl;
	RitzMatrix(evec,Nk);
	t1=usecond()/1e6;
	std::cout<<GridLogMessage <<"IRL::RitzMatrix: "<<t1-t0<< "seconds"<<std::endl; t0=t1;
	for(int k=0; k<Nk; ++k){
//	std:: cout<<GridLogMessage <<"eval " << k << " " <<eval[k] << std::endl;
//	std:: cout<<GridLogMessage <<"lme " << k << " " << lme[k] << std::endl;
	}

	// Restarting loop begins
	for(int iter = 0; iter<Niter; ++iter){
	  
	  std::cout<<GridLogMessage<<"\n Restart iteration = "<< iter << std::endl;
	  
	  // 
	  // Rudy does a sort first which looks very different. Getting fed up with sorting out the algo defs.
	  // We loop over 
	  //
	  OrthoTime=0.;
	  for(int k=Nk; k<Nm; ++k) step(eval,lme,evec,f,Nm,k,evec_offset);
	  t1=usecond()/1e6;
	  std::cout<<GridLogMessage <<"IRL:: "<<Np <<" steps: "<<t1-t0<< "seconds"<<std::endl; t0=t1;
	  std::cout<<GridLogMessage <<"IRL::Initial steps:OrthoTime "<<OrthoTime<< "seconds"<<std::endl;
	  f *= lme[Nm-1];
	  
	  RitzMatrix(evec,k2);
	  t1=usecond()/1e6;
	  std::cout<<GridLogMessage <<"IRL:: RitzMatrix: "<<t1-t0<< "seconds"<<std::endl; t0=t1;
	  
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
	  evec.rotate(Qt,k1-1,k2+1,0,Nm,Nm,evec_offset);
	  
	  t1=usecond()/1e6;
	  std::cout<<GridLogMessage <<"IRL::QR rotation: "<<t1-t0<< "seconds"<<std::endl; t0=t1;
	  fflush(stdout);
	  
	  // Compressed vector f and beta(k2)
	  f *= Qt[Nm-1+Nm*(k2-1)];
	  f += lme[k2-1] * evec.get(k2 + evec_offset);
	  beta_k = norm2(f);
	  beta_k = sqrt(beta_k);
	  std::cout<<GridLogMessage<<" beta(k) = "<<beta_k<<std::endl;
	  
	  RealD betar = 1.0/beta_k;
	  evec.put(k2 + evec_offset, betar * f);
	  lme[k2-1] = beta_k;
	  
	  // Convergence test
	  for(int k=0; k<Nm; ++k){    
	    eval2[k] = eval[k];
	    lme2[k] = lme[k];
	  }
	  setUnit_Qt(Nm,Qt);
	  diagonalize(eval2,lme2,Nk,Nm,Qt,grid);
	  t1=usecond()/1e6;
	  std::cout<<GridLogMessage <<"IRL::diagonalize: "<<t1-t0<< "seconds"<<std::endl; t0=t1;
	  
	  
	  Nconv = 0;
	  
	  if (iter >= Nminres) {
	    std::cout << GridLogMessage << "Rotation to test convergence " << std::endl;
	    
	    FieldHP ev0_orig(grid);
	    ev0_orig = evec.get(0 + evec_offset);
	    
	    evec.rotate(Qt,0,Nk,0,Nk,Nm,evec_offset);
	    
	    {
	      std::cout << GridLogMessage << "Test convergence" << std::endl;
	      FieldHP B(grid);
	      
	      for(int j = 0; j<Nk; ++j){
		B=evec.get(j + evec_offset);
		B.checkerboard = evec.get(0 + evec_offset).checkerboard;
		//std::cout << GridLogMessage << "Checkerboard: " << B.checkerboard << " norm2 = " << norm2(B) << std::endl;
		
		/*{
		  auto res = B._odata[0];
		  std::cout << GridLogMessage << " ev = " << res << std::endl;
		  }*/
		FieldHP tmp(B);
		Field   tmpLP(evec._v[0]._grid);
		Field   vLP(tmpLP);
		_proj(B,v);
		precisionChange(vLP,v);
		if (!test_conv_poly)
		  _Linop.HermOp(vLP,tmpLP);
		else
		  _poly(_Linop,vLP,tmpLP);      // 3. wk:=Avk−βkv_{k−1}
		precisionChange(tmp,tmpLP);
		_proj(tmp,v);
		
		RealD vnum = real(innerProduct(B,v)); // HermOp.
		RealD vden = norm2(B);
		RealD vv0 = norm2(v);
		eval2[j] = vnum/vden;
		v -= eval2[j]*B;
		RealD vv = norm2(v);
		std::string xtr;
		if (test_conv_poly) {
		  vv /= ::pow(eval2[j],2.0);
		  xtr = "/ eval[i]^2 ";
		}
		std::cout.precision(13);
		std::cout<<GridLogMessage << "[" << std::setw(3)<< std::setiosflags(std::ios_base::right) <<j<<"] "
			 <<"eval = "<<std::setw(25)<< std::setiosflags(std::ios_base::left)<< eval2[j]
			 <<" |H B[i] - eval[i]B[i]|^2 " << xtr << std::setw(25)<< std::setiosflags(std::ios_base::right)<< vv
			 <<" "<< vnum/(sqrt(vden)*sqrt(vv0))
			 << " norm(B["<<j<<"])="<< vden <<std::endl;
		
		// change the criteria as evals are supposed to be sorted, all evals smaller(larger) than Nstop should have converged
		if((vv<eresid*eresid) && (j == Nconv) ){
		  Iconv[Nconv] = j;
		  ++Nconv;
		}
	      }
	      
	      // test if we converged, if so, terminate
	      t1=usecond()/1e6;
	      std::cout<<GridLogMessage <<"IRL::convergence testing: "<<t1-t0<< "seconds"<<std::endl; t0=t1;
	      
	      std::cout<<GridLogMessage<<" #modes converged: "<<Nconv<<std::endl;
	      
	      if( Nconv>=Nstop ){
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
		evec.rotate(QtI,0,Nk,0,Nk,Nm,evec_offset);
		
		axpy(ev0_orig,-1.0,evec.get(0 + evec_offset),ev0_orig);
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
       // Sorting

       eval.resize(Nconv);

       for(int i=0; i<Nconv; ++i)
         //eval[i] = eval2[Iconv[i]];
	 eval[i] = eval2[i]; // for now just take the lowest Nconv, should be fine the way Lanc converges
       
       {
	 
	 // test
	 for (int j=0;j<Nconv;j++) {
	   std::cout<<GridLogMessage << " |e[" << j << "]|^2 = " << norm2(evec.get(j + evec_offset)) << std::endl;
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

