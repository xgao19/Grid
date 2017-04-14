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
// eliminate temorary vector in calc()

namespace Grid {

template<typename Coeff_t,typename Field>
class BlockedGrid {
public:
  GridBase* _grid;
  
  std::vector<int> _bs; // block size
  std::vector<int> _nb; // number of blocks
  std::vector<int> _l;  // local dimensions irrespective of cb
  std::vector<int> _l_cb;  // local dimensions of checkerboarded vector
  std::vector<int> _bs_cb; // block size in checkerboarded vector
  std::vector<int> _l_cb_o;  // local dimensions in terms of cb and simd o-sites
  std::vector<int> _bs_cb_o; // block size in terms of cb and simd o-sites

  int _nd, _blocks, _cf_size, _cf_block_size, _o_sites, _o_block_sites;
  
  BlockedGrid(GridBase* grid, const std::vector<int>& block_size) :
    _grid(grid), _bs(block_size), _nd((int)_bs.size()), 
      _nb(block_size), _l(block_size), _bs_cb_o(block_size),
      _l_cb_o(block_size), _bs_cb(block_size) {

    _blocks = 1;
    _l = grid->FullDimensions();
    _l_cb = grid->LocalDimensions();

    _cf_size = 1;
    _o_sites = 1;

    for (int i=0;i<_nd;i++) {
      _l[i] /= grid->_processors[i];

      assert(!(_l[i] % _bs[i])); // lattice must accommodate choice of blocksize

      int r = _l[i] / _l_cb[i];
      assert(!(_bs[i] % r)); // checkerboarding must accommodate choice of blocksize
      _bs_cb[i] = _bs[i] / r;
      _l_cb_o[i] = _l_cb[i] / _grid->_simd_layout[i]; // this is already guaranteed to divide
      _o_sites *= _l_cb_o[i];
      assert(!(_bs_cb[i] % _grid->_simd_layout[i])); // simd must accommodate choice of blocksize
      _bs_cb_o[i] = _bs_cb[i] / _grid->_simd_layout[i];
      _nb[i] = _l[i] / _bs[i];
      _blocks *= _nb[i];
      _cf_size *= _l[i];
    }

    _cf_size *= 12 / 2;
    _cf_block_size = _cf_size / _blocks;
    _o_block_sites = _o_sites / _blocks;

    std::cout << GridLogMessage << "BlockedGrid:" << std::endl;
    std::cout << GridLogMessage << " _l     = " << _l << std::endl;
    std::cout << GridLogMessage << " _l_cb     = " << _l_cb << std::endl;
    std::cout << GridLogMessage << " _l_cb_o     = " << _l_cb_o << std::endl;
    std::cout << GridLogMessage << " _bs    = " << _bs << std::endl;
    std::cout << GridLogMessage << " _bs_cb    = " << _bs_cb << std::endl;
    std::cout << GridLogMessage << " _bs_cb_o    = " << _bs_cb_o << std::endl;
    std::cout << GridLogMessage << " _nb    = " << _nb << std::endl;
    std::cout << GridLogMessage << " _blocks = " << _blocks << std::endl;
    std::cout << GridLogMessage << " sizeof(Coeff_t) = " << sizeof(Coeff_t) << std::endl;
    std::cout << GridLogMessage << " _cf_size = " << _cf_size << std::endl;
    std::cout << GridLogMessage << " _cf_block_size = " << _cf_block_size << std::endl;
    std::cout << GridLogMessage << " _o_sites = " << _o_sites << std::endl;
    std::cout << GridLogMessage << " _o_block_sites = " << _o_block_sites << std::endl;
    std::cout << GridLogMessage << " _grid->oSites() = " << _grid->oSites() << std::endl;

    //    _grid->Barrier();
    //abort();
  }

    void block_to_coor(int b, std::vector<int>& x0) {

      std::vector<int> bcoor;
      bcoor.resize(_nd);
      x0.resize(_nd);
      Lexicographic::CoorFromIndex(bcoor,b,_nb);
      int i;

      for (i=0;i<_nd;i++) {
	x0[i] = bcoor[i]*_bs_cb_o[i];
      }

      //std::cout << GridLogMessage << "Map block b -> " << x0 << " -- " << x1 << std::endl;

    }

    int block_site_to_o_site(const std::vector<int>& x0, int i) {
      std::vector<int> coor;  coor.resize(_nd);
      Lexicographic::CoorFromIndex(coor,i,_bs_cb_o);
      for (int j=0;j<_nd;j++)
	coor[j] += x0[j];
      Lexicographic::IndexFromCoor(coor,i,_l_cb_o);
      return i;
    }

    ComplexD block_sp(int b, const Field& x, const Field& y) {

      std::vector<int> x0;
      block_to_coor(b,x0);

      //std::cout << GridLogMessage << "Coor x0 = " << x0 << std::endl;
      
      ComplexD ret = 0.0;

      for (int i=0;i<_o_block_sites;i++) { // only odd sites
	int ss = block_site_to_o_site(x0,i);
	//std::cout << GridLogMessage << i << "/" << _o_block_sites << " -> " << ss << std::endl;
	ret += Reduce(TensorRemove(innerProduct(x._odata[ss],y._odata[ss])));
      }
      return ret;
    }

    void block_caxpy(int b, Field& ret, const Coeff_t& a, const Field& x, const Field& y) {

      std::vector<int> x0;
      block_to_coor(b,x0);
      
      for (int i=0;i<_o_block_sites;i++) { // only odd sites
	int ss = block_site_to_o_site(x0,i);
	ret._odata[ss] = a*x._odata[ss] + y._odata[ss];
      }

    }

    void block_cscale(int b, const Coeff_t& a, Field& ret) {

      std::vector<int> x0;
      block_to_coor(b,x0);
      
      for (int i=0;i<_o_block_sites;i++) { // only odd sites
	int ss = block_site_to_o_site(x0,i);
	ret._odata[ss] = a * ret._odata[ss];
      }

    }

};
  
template<class Field>
class BlockedFieldVector {
 public:
  int _Nm,  // number of total vectors
    _Nfull; // number of vectors kept in full precision

  typedef typename Field::scalar_type Coeff_t;
  typedef typename Field::vector_object vobj;
  typedef typename vobj::scalar_object sobj;

  BlockedGrid<Coeff_t,Field> _bgrid;
  
  std::vector<Field> _v; // _Nfull vectors
  std::vector< Coeff_t > _c;

  bool _full_locked;
  
  //public:

  BlockedFieldVector(int Nm,GridBase* value,int Nfull,const std::vector<int>& block_size) : 
  _Nfull(Nfull), _Nm(Nm), _v(Nfull,value), _bgrid(value,block_size), _full_locked(false) {

    _c.resize(_Nm*_Nfull*_bgrid._blocks);
#pragma omp parallel for
    for (int64_t i=0;i<_c.size();i++)
      _c[i] = 0.0;

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

#define cidx(i,b,j) ((int64_t)b + (int64_t)_bgrid._blocks * (int64_t)(j + _Nfull*i))
  void set_coef(int i, int b, int j, const Coeff_t& v) {
    _c[ cidx(i,b,j) ] = v;
  }

  void lock_in_full_vectors() {

    assert(!_full_locked);
    _full_locked = true;

    GridStopWatch sw;
    sw.Start();
    // orthogonalize local blocks and create coefficients for first Nfull vectors
#pragma omp parallel for
    for (int b=0;b<_bgrid._blocks;b++) {
      for (int i=0;i<_Nfull;i++) {
	// |i> -= <j|i> |j>
	for (int j=0;j<i;j++) {
	  Coeff_t v = (Coeff_t)_bgrid.block_sp(b,_v[j],_v[i]);
	  set_coef(i,b,j,v);
	  _bgrid.block_caxpy(b,_v[i],-v,_v[j],_v[i]);
	}

	Coeff_t nrm = (Coeff_t)_bgrid.block_sp(b,_v[i],_v[i]);
	set_coef(i,b,i,sqrt(nrm));
	
	_bgrid.block_cscale(b,1.0 / sqrt(nrm),_v[i]);
      }
    }
    sw.Stop();
    std::cout << GridLogMessage << "Gram-Schmidt to create blocked basis took " << sw.Elapsed() << std::endl;

#if 0
    // test blocked sp
    Coeff_t sm = 0.0;
    for (int b=0;b<_bgrid._blocks;b++) {
      sm += (Coeff_t)_bgrid.block_sp(b,_v[_Nfull-1],_v[_Nfull-1]);
    }
    _bgrid._grid->GlobalSum(sm);
    Coeff_t tot = (Coeff_t)innerProduct(_v[_Nfull-1],_v[_Nfull-1]);
    std::cout << GridLogMessage << " sm = " << sm << " tot = " << tot << std::endl;

    // test basis
    for (int j=0;j<_Nfull;j++) {
      for (int b=0;b<_bgrid._blocks;b++) {
	auto nrm2 = _bgrid.block_sp(b,_v[j],_v[j]);
	std::cout << GridLogMessage << "TEST b = " << b << " j = " << j
		  << " nrm2(_v) = " << nrm2 
		  << " ortho[j,1] = " << _bgrid.block_sp(b,_v[j],_v[1]) << std::endl;
	
      }

      _bgrid._grid->Barrier();
      abort();
    }
#endif

  }

  Field get_blocked(int i) {

    Field ret(_bgrid._grid);
    ret = zero;
    ret.checkerboard = _v[0].checkerboard;

#pragma omp parallel for
    for (int b=0;b<_bgrid._blocks;b++) {
      for (int j=0;j<_Nfull;j++)
	_bgrid.block_caxpy(b,ret,_c[ cidx(i,b,j) ],_v[j],ret);
    }
    
    return ret;
  }

  void put_blocked(int i, const Field& rhs) {

#pragma omp parallel for
    for (int b=0;b<_bgrid._blocks;b++) {
      for (int j=0;j<_Nfull;j++) {
	// |rhs> -= <j|rhs> |j>
	Coeff_t v = (Coeff_t)_bgrid.block_sp(b,_v[j],rhs);
	set_coef(i,b,j,v);
      }
    }

#if 0
    std::cout << GridLogMessage << "------------------" << std::endl;
    for (int b=0;b<_bgrid._blocks;b++) {
      for (int j=0;j<_Nfull;j++) {
    std::cout << GridLogMessage << " c[" << i << ", " << b << ", " << j << "] = " << _c[ cidx(i,b,j) ] << std::endl;
  }
  }

    std::cout << GridLogMessage << "------------------" << std::endl;

    if (i==31) {
    _bgrid._grid->Barrier();
    abort();
    }
#endif

  }

  Field get(int i) {
    if (!_full_locked) {
      assert(i < _Nfull);
      return _v[i];
    } else {
      return get_blocked(i);
    }
  }

  void put(int i, const Field& v) {
    //std::cout << GridLogMessage << "::put(" << i << ")\n";

    if (!_full_locked && i >= _Nfull) {
      // lock in smallest vectors so we can build on them
      Field test = _v[_Nfull - 1];
      lock_in_full_vectors();
      axpy(test,-1.0,get_blocked(_Nfull - 1),test);
      std::cout << GridLogMessage << "Error of lock_in_full_vectors: " << norm2(test) << std::endl;
    }

    if (!_full_locked) {
      assert(i < _Nfull);
      _v[i] = v;
    } else {
      put_blocked(i,v);

      Field test = get_blocked(i);
      RealD nrm2b = norm2(test);
      axpy(test,-1.0,v,test);
      std::cout << GridLogMessage << "Error of vector: " << norm2(test) << " nrm2 = " << norm2(v) << " vs " << nrm2b << std::endl;
    }
  }

  void rotate(DenseVector<RealD>& Qt,int j0, int j1, int k0,int k1) {
    GridBase* grid = _v[0]._grid;
    
    if (!_full_locked) {
      
#pragma omp parallel
      {
	std::vector < vobj > B(_Nm);
	
#pragma omp for
	for(int ss=0;ss < grid->oSites();ss++){
	  for(int j=j0; j<j1; ++j) B[j]=0.;
	  
	  for(int j=j0; j<j1; ++j){
	    for(int k=k0; k<k1; ++k){
	      B[j] +=Qt[k+_Nm*j] * _v[k]._odata[ss];
	    }
	  }
	  for(int j=j0; j<j1; ++j){
	    _v[j]._odata[ss] = B[j];
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
        std::vector<Coeff_t> c0;
	c0.resize(_Nfull*_Nm);

#pragma omp for
        for (int b=0;b<_bgrid._blocks;b++) {
          for (int l=0;l<_Nfull;l++) {
	    for(int j=j0; j<j1; ++j){
	      Coeff_t& cc = c0[l + _Nfull*j];
	      cc = 0.0;
	      for(int k=k0; k<k1; ++k){
		cc +=Qt[k+_Nm*j] * _c[ cidx(k,b,l) ];
	      }
	    }
          }
	  for (int l=0;l<_Nfull;l++) {
	    for(int j=j0; j<j1; ++j){
	      _c[ cidx(j,b,l) ] = c0[l + _Nfull*j];
	    }
	  }
	}
      }
      
    }
  }

  size_t size() const {
    return _Nm;
  }

  void sort(DenseVector<RealD>& sort_vals, int Nsort) {
    // TODO
  }


  // zlib's crc32 gets 0.4 GB/s on KNL single thread
  // below gets 4.8 GB/s
  uint32_t crc32_threaded(unsigned char* data, int64_t len, uint32_t previousCrc32 = 0) {

    return crc32(previousCrc32,data,len);

    /*
    std::vector<uint32_t> partials;
    int64_t len_part;

#pragma omp parallel shared(partials,len_part)
    {
      int threads = omp_get_num_threads();
      int thread  = omp_get_thread_num();

#pragma omp single
      {
	partials.resize(threads);
	assert(len % threads == 0); // for now 64 divides all our data, easy to generalize
	len_part = len / threads;
      }

#pragma omp barrier

      partials[thread] = crc32(previousCrc32,data + len_part * thread,len_part);
    }

    uint32_t ret = crc32_combine(partials[0],partials[1],len_part);
    for (int i=2;i<(int)partials.size();i++)
      ret = crc32_combine(ret,partials[i],len_part);
    return ret;
    */
      
  }

  int get_bfm_index( int* pos, int co, int* s ) {
    
    int ls = s[0];
    int NtHalf = s[4] / 2;
    int simd_coor = pos[4] / NtHalf;
    int regu_coor = (pos[1] + s[1] * (pos[2] + s[2] * ( pos[3] + s[3] * (pos[4] % NtHalf) ) )) / 2;
     
    return regu_coor * ls * 48 + pos[0] * 48 + co * 4 + simd_coor * 2;
  }

  void read_argonne(const char* dir, std::vector<int>& nodes) {
    // this is slow code to read the argonne file format for debugging purposes
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
      assert(f);
      for (int l=0;l<3+slot;l++)
	fgets(buf,sizeof(buf),f);
      uint32_t crc_exp = strtol(buf, NULL, 16);
      fclose(f);

      // load one slot vector
      sprintf(buf,"%s/%2.2d/%10.10d",dir,slot/nperdir,slot);
      f = fopen(buf,"rb");
      assert(f);
      
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
  }
};

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

    int Nstop;   // Number of evecs checked for convergence
    int Nk;      // Number of converged sought
    int Np;      // Np -- Number of spare vecs in kryloc space
    int Nm;      // Nm -- total number of vectors


    RealD OrthoTime;

    RealD eresid;

    SortEigen<Field> _sort;

    LinearOperatorBase<Field> &_Linop;

    OperatorFunction<Field>   &_poly;

    /////////////////////////
    // Constructor
    /////////////////////////
    void init(void){};
    void Abort(int ff, DenseVector<RealD> &evals,  DenseVector<DenseVector<RealD> > &evecs);

    ImplicitlyRestartedLanczos(
				LinearOperatorBase<Field> &Linop, // op
			       OperatorFunction<Field> & poly,   // polynmial
			       int _Nstop, // sought vecs
			       int _Nk, // sought vecs
			       int _Nm, // spare vecs
			       RealD _eresid, // resid in lmdue deficit 
			       int _Niter) : // Max iterations
      _Linop(Linop),
      _poly(poly),
      Nstop(_Nstop),
      Nk(_Nk),
      Nm(_Nm),
      eresid(_eresid),
      Niter(_Niter)
    { 
      Np = Nm-Nk; assert(Np>0);
    };

    ImplicitlyRestartedLanczos(
				LinearOperatorBase<Field> &Linop, // op
			       OperatorFunction<Field> & poly,   // polynmial
			       int _Nk, // sought vecs
			       int _Nm, // spare vecs
			       RealD _eresid, // resid in lmdue deficit 
			       int _Niter) : // Max iterations
      _Linop(Linop),
      _poly(poly),
      Nstop(_Nk),
      Nk(_Nk),
      Nm(_Nm),
      eresid(_eresid),
      Niter(_Niter)
    { 
      Np = Nm-Nk; assert(Np>0);
    };

    /////////////////////////
    // Sanity checked this routine (step) against Saad.
    /////////////////////////
    void RitzMatrix(BlockedFieldVector<Field>& evec,int k){

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
	      BlockedFieldVector<Field>& evec,
	      Field& w,int Nm,int k)
    {
      assert( k< Nm );

      Field evec_k = evec.get(k);

      _poly(_Linop,evec_k,w);      // 3. wk:=Avk−βkv_{k−1}

#if 0
      // Should we adopt the compression?
      if (k < Nm - 1) {
	evec.put(k+1,w);
	w = evec.get(k+1);
      }
#endif

      if(k>0){
	w -= lme[k-1] * evec.get(k-1);
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

      if (k>0) { 
	orthogonalize(w,evec,k); // orthonormalise
      }

      if(k < Nm-1) { 
	evec.put(k+1, w);
	//w = evec.get(k+1);  // adopt compression for w?
      }

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
    static RealD normalise(Field& v) 
    {
      RealD nn = norm2(v);
      nn = sqrt(nn);
      v = v * (1.0/nn);
      return nn;
    }

    void orthogonalize(Field& w,
		       BlockedFieldVector<Field>& evec,
		       int k)
    {
      double t0=-usecond()/1e6;
      typedef typename Field::scalar_type MyComplex;
      MyComplex ip;

#if 0
	for(int j=0; j<k; ++j){
	  normalise(evec[j]);
	  for(int i=0;i<j;i++){
	    ip = innerProduct(evec[i],evec[j]); // are the evecs normalised? ; this assumes so.
	    evec[j] = evec[j] - ip *evec[i];
	  }
	}
#endif

      for(int j=0; j<k; ++j){
        Field evec_j = evec.get(j);
	ip = innerProduct(evec_j,w); // are the evecs normalised? ; this assumes so.
	w = w - ip * evec_j;
      }
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

// alternate implementation for minimizing memory usage.  May affect the performance
    void calc(DenseVector<RealD>& eval,
	      BlockedFieldVector<Field>& evec,
	      const Field& src,
	      int& Nconv)
      {

	GridBase *grid = evec.get(0)._grid;
	assert(grid == src._grid);

	std::cout<<GridLogMessage << " -- Nk = " << Nk << " Np = "<< Np << std::endl;
	std::cout<<GridLogMessage << " -- Nm = " << Nm << std::endl;
	std::cout<<GridLogMessage << " -- size of eval   = " << eval.size() << std::endl;
	std::cout<<GridLogMessage << " -- size of evec  = " << evec.size() << std::endl;
	
	assert(Nm == evec.size() && Nm == eval.size());
	
	DenseVector<RealD> lme(Nm);  
	DenseVector<RealD> lme2(Nm);
	DenseVector<RealD> eval2(Nm);
	DenseVector<RealD> Qt(Nm*Nm);
	DenseVector<int>   Iconv(Nm);

#if (!defined MEM_SAVE )
	DenseVector<Field>  B(Nm,grid); // waste of space replicating
#endif
	
	Field f(grid);
	Field v(grid);
  
	int k1 = 1;
	int k2 = Nk;

	Nconv = 0;

	RealD beta_k;
  
	// Set initial vector
	// (uniform vector) Why not src??
	//	evec[0] = 1.0;
	Field src_n=src;
	normalise(src_n);
	evec.put(0,src_n);
	std:: cout<<GridLogMessage <<"norm2(src)= " << norm2(src)<<std::endl;
// << src._grid  << std::endl;
	std:: cout<<GridLogMessage <<"norm2(evec[0])= " << norm2(evec.get(0)) <<std::endl;
// << evec[0]._grid << std::endl;
	
	// Initial Nk steps
	OrthoTime=0.;
	double t0=usecond()/1e6;
	for(int k=0; k<Nk; ++k) step(eval,lme,evec,f,Nm,k);
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
	  for(int k=Nk; k<Nm; ++k) step(eval,lme,evec,f,Nm,k);
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

#if 0
	if (grid->ThisRank() == 0){
	  printf("Q:\n");
	  for (int j=0;j<Nm; j++) {
	    RealD n2 = 0.0;
	    RealD n8 = 0.0;

	    for(int k=0; k<Nm ; ++k){
	      RealD a = ::fabs(Qt[k+Nm*j]);
	      //printf("%5.5g ",a);
	      n2 += ::pow(a,2.0);
	      n8 += ::pow(a,8.0);
	    }

	    printf("%d = %g\n",j,::pow(n8,1.0/8.0) / ::pow(n2,1.0/2.0));
	  }
	}
#endif

	assert(k2<Nm);
	assert(k1>0);
	evec.rotate(Qt,k1-1,k2+1,0,Nm);

	t1=usecond()/1e6;
	std::cout<<GridLogMessage <<"IRL::QR rotation: "<<t1-t0<< "seconds"<<std::endl; t0=t1;
	fflush(stdout);

	  // Compressed vector f and beta(k2)
	  f *= Qt[Nm-1+Nm*(k2-1)];
	  f += lme[k2-1] * evec.get(k2);
	  beta_k = norm2(f);
	  beta_k = sqrt(beta_k);
	  std::cout<<GridLogMessage<<" beta(k) = "<<beta_k<<std::endl;

	  RealD betar = 1.0/beta_k;
	  evec.put(k2, betar * f);
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


	std::cout << GridLogMessage << "Rotation to test convergence " << std::endl;

	{
	  Field ev0_orig(grid);
	  ev0_orig = evec.get(0);
	  
	  evec.rotate(Qt,0,Nk,0,Nk);
	  
	  {
	    std::cout << GridLogMessage << "Test convergence" << std::endl;
	    Field B(grid);
	    
	    for(int j = 0; j<Nk; ++j){
	      B=evec.get(j);
	      B.checkerboard = evec.get(0).checkerboard;

	      /*{
		auto res = B._odata[0];
		std::cout << GridLogMessage << " ev = " << res << std::endl;
		}*/
	      _Linop.HermOp(B,v);
	      
	      RealD vnum = real(innerProduct(B,v)); // HermOp.
	      RealD vden = norm2(B);
	      RealD vv0 = norm2(v);
	      eval2[j] = vnum/vden;
	      v -= eval2[j]*B;
	      RealD vv = norm2(v);
	      
	      std::cout.precision(13);
	      std::cout<<GridLogMessage << "[" << std::setw(3)<< std::setiosflags(std::ios_base::right) <<j<<"] "
		       <<"eval = "<<std::setw(25)<< std::setiosflags(std::ios_base::left)<< eval2[j]
		       <<"|H B[i] - eval[i]B[i]|^2 "<< std::setw(25)<< std::setiosflags(std::ios_base::right)<< vv
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
	      evec.rotate(QtI,0,Nk,0,Nk);
	      
	      axpy(ev0_orig,-1.0,evec.get(0),ev0_orig);
	      std::cout << GridLogMessage << "Rotation done (in " << timeInv.Elapsed() << " = " << timeInv.useconds() << " us" <<
		", error = " << res_check_rotate_inverse << 
		"); | evec[0] - evec[0]_orig | = " << ::sqrt(norm2(ev0_orig)) << std::endl;
	    }
	  }
	}
	} // end of iter loop
	
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
	   std::cout<<GridLogMessage << " |e[" << j << "]|^2 = " << norm2(evec.get(j)) << std::endl;
	 }
       }

       //_sort.push(eval,evec,Nconv);
       evec.sort(eval,Nconv);
       
       std::cout<<GridLogMessage << "\n Converged\n Summary :\n";
       std::cout<<GridLogMessage << " -- Iterations  = "<< Nconv  << "\n";
       std::cout<<GridLogMessage << " -- beta(k)     = "<< beta_k << "\n";
       std::cout<<GridLogMessage << " -- Nconv       = "<< Nconv  << "\n";
     }

    /////////////////////////////////////////////////
    // Adapted from Rudy's lanczos factor routine
    /////////////////////////////////////////////////
    int Lanczos_Factor(int start, int end,  int cont,
		       DenseVector<Field> & bq, 
		       Field &bf,
		       DenseMatrix<RealD> &H){
      
      GridBase *grid = bq[0]._grid;

      RealD beta;  
      RealD sqbt;  
      RealD alpha;

      for(int i=start;i<Nm;i++){
	for(int j=start;j<Nm;j++){
	  H[i][j]=0.0;
	}
      }

      std::cout<<GridLogMessage<<"Lanczos_Factor start/end " <<start <<"/"<<end<<std::endl;

      // Starting from scratch, bq[0] contains a random vector and |bq[0]| = 1
      int first;
      if(start == 0){

	std::cout<<GridLogMessage << "start == 0\n"; //TESTING

	_poly(_Linop,bq[0],bf);

	alpha = real(innerProduct(bq[0],bf));//alpha =  bq[0]^dag A bq[0]

	std::cout<<GridLogMessage << "alpha = " << alpha << std::endl;
	
	bf = bf - alpha * bq[0];  //bf =  A bq[0] - alpha bq[0]

	H[0][0]=alpha;

	std::cout<<GridLogMessage << "Set H(0,0) to " << H[0][0] << std::endl;

	first = 1;

      } else {

	first = start;

      }

      // I think start==0 and cont==zero are the same. Test this
      // If so I can drop "cont" parameter?
      if( cont ) assert(start!=0);

      if( start==0 ) assert(cont!=0);

      if( cont){

	beta = 0;sqbt = 0;

	std::cout<<GridLogMessage << "cont is true so setting beta to zero\n";

      }	else {

	beta = norm2(bf);
	sqbt = sqrt(beta);

	std::cout<<GridLogMessage << "beta = " << beta << std::endl;
      }

      for(int j=first;j<end;j++){

	std::cout<<GridLogMessage << "Factor j " << j <<std::endl;

	if(cont){ // switches to factoring; understand start!=0 and initial bf value is right.
	  bq[j] = bf; cont = false;
	}else{
	  bq[j] = (1.0/sqbt)*bf ;

	  H[j][j-1]=H[j-1][j] = sqbt;
	}

	_poly(_Linop,bq[j],bf);

	bf = bf - (1.0/sqbt)*bq[j-1]; 	       //bf = A bq[j] - beta bq[j-1] // PAB this comment was incorrect in beta term??

	alpha = real(innerProduct(bq[j],bf));  //alpha = bq[j]^dag A bq[j]

	bf = bf - alpha*bq[j];                 //bf = A bq[j] - beta bq[j-1] - alpha bq[j]
	RealD fnorm = norm2(bf);

	RealD bck = sqrt( real( conjugate(alpha)*alpha ) + beta );

	beta = fnorm;
	sqbt = sqrt(beta);
	std::cout<<GridLogMessage << "alpha = " << alpha << " fnorm = " << fnorm << '\n';

	///Iterative refinement of orthogonality V = [ bq[0]  bq[1]  ...  bq[M] ]
	int re = 0;
	// FIXME undefined params; how set in Rudy's code
	int ref =0;
	Real rho = 1.0e-8;

	while( re == ref || (sqbt < rho * bck && re < 5) ){

	  Field tmp2(grid);
	  Field tmp1(grid);

	  //bex = V^dag bf
	  DenseVector<ComplexD> bex(j+1);
	  for(int k=0;k<j+1;k++){
	    bex[k] = innerProduct(bq[k],bf);
	  }
	  
	  zero_fermion(tmp2);
	  //tmp2 = V s
	  for(int l=0;l<j+1;l++){
	    RealD nrm = norm2(bq[l]);
	    axpy(tmp1,0.0,bq[l],bq[l]); scale(tmp1,bex[l]); 	//tmp1 = V[j] bex[j]
	    axpy(tmp2,1.0,tmp2,tmp1);					//tmp2 += V[j] bex[j]
	  }

	  //bf = bf - V V^dag bf.   Subtracting off any component in span { V[j] } 
	  RealD btc = axpy_norm(bf,-1.0,tmp2,bf);
	  alpha = alpha + real(bex[j]);	      sqbt = sqrt(real(btc));	      
	  // FIXME is alpha real in RUDY's code?
	  RealD nmbex = 0;for(int k=0;k<j+1;k++){nmbex = nmbex + real( conjugate(bex[k])*bex[k]  );}
	  bck = sqrt( nmbex );
	  re++;
	}
	std::cout<<GridLogMessage << "Iteratively refined orthogonality, changes alpha\n";
	if(re > 1) std::cout<<GridLogMessage << "orthagonality refined " << re << " times" <<std::endl;
	H[j][j]=alpha;
      }

      return end;
    }

    void EigenSort(DenseVector<double> evals,
		   DenseVector<Field>  evecs){
      int N= evals.size();
      _sort.push(evals,evecs, evals.size(),N);
    }

    void ImplicitRestart(int TM, DenseVector<RealD> &evals,  DenseVector<DenseVector<RealD> > &evecs, DenseVector<Field> &bq, Field &bf, int cont)
    {
      std::cout<<GridLogMessage << "ImplicitRestart begin. Eigensort starting\n";

      DenseMatrix<RealD> H; Resize(H,Nm,Nm);

#ifndef USE_LAPACK
      EigenSort(evals, evecs);
#endif

      ///Assign shifts
      int K=Nk;
      int M=Nm;
      int P=Np;
      int converged=0;
      if(K - converged < 4) P = (M - K-1); //one
      //      DenseVector<RealD> shifts(P + shift_extra.size());
      DenseVector<RealD> shifts(P);
      for(int k = 0; k < P; ++k)
	shifts[k] = evals[k]; 

      /// Shift to form a new H and q
      DenseMatrix<RealD> Q; Resize(Q,TM,TM);
      Unity(Q);
      Shift(Q, shifts); // H is implicitly passed in in Rudy's Shift routine

      int ff = K;

      /// Shifted H defines a new K step Arnoldi factorization
      RealD  beta = H[ff][ff-1]; 
      RealD  sig  = Q[TM - 1][ff - 1];
      std::cout<<GridLogMessage << "beta = " << beta << " sig = " << real(sig) <<std::endl;

      std::cout<<GridLogMessage << "TM = " << TM << " ";
      std::cout<<GridLogMessage << norm2(bq[0]) << " -- before" <<std::endl;

      /// q -> q Q
      times_real(bq, Q, TM);

      std::cout<<GridLogMessage << norm2(bq[0]) << " -- after " << ff <<std::endl;
      bf =  beta* bq[ff] + sig* bf;

      /// Do the rest of the factorization
      ff = Lanczos_Factor(ff, M,cont,bq,bf,H);
      
      if(ff < M)
	Abort(ff, evals, evecs);
    }

///Run the Eigensolver
    void Run(int cont, DenseVector<Field> &bq, Field &bf, DenseVector<DenseVector<RealD> > & evecs,DenseVector<RealD> &evals)
    {
      init();

      int M=Nm;

      DenseMatrix<RealD> H; Resize(H,Nm,Nm);
      Resize(evals,Nm);
      Resize(evecs,Nm);

      int ff = Lanczos_Factor(0, M, cont, bq,bf,H); // 0--M to begin with

      if(ff < M) {
	std::cout<<GridLogMessage << "Krylov: aborting ff "<<ff <<" "<<M<<std::endl;
	abort(); // Why would this happen?
      }

      int itcount = 0;
      bool stop = false;

      for(int it = 0; it < Niter && (converged < Nk); ++it) {

	std::cout<<GridLogMessage << "Krylov: Iteration --> " << it << std::endl;
	int lock_num = lock ? converged : 0;
	DenseVector<RealD> tevals(M - lock_num );
	DenseMatrix<RealD> tevecs; Resize(tevecs,M - lock_num,M - lock_num);
	  
	//check residual of polynominal 
	TestConv(H,M, tevals, tevecs);

	if(converged >= Nk)
	    break;

	ImplicitRestart(ff, tevals,tevecs,H);
      }
      Wilkinson<RealD>(H, evals, evecs, small); 
      //      Check();

      std::cout<<GridLogMessage << "Done  "<<std::endl;

    }

   ///H - shift I = QR; H = Q* H Q
    void Shift(DenseMatrix<RealD> & H,DenseMatrix<RealD> &Q, DenseVector<RealD> shifts) {
      
      int P; Size(shifts,P);
      int M; SizeSquare(Q,M);

      Unity(Q);

      int lock_num = lock ? converged : 0;

      RealD t_Househoulder_vector(0.0);
      RealD t_Househoulder_mult(0.0);

      for(int i=0;i<P;i++){

	RealD x, y, z;
	DenseVector<RealD> ck(3), v(3);
	  
	x = H[lock_num+0][lock_num+0]-shifts[i];
	y = H[lock_num+1][lock_num+0];
	ck[0] = x; ck[1] = y; ck[2] = 0; 

	normalise(ck);	///Normalization cancels in PHP anyway
	RealD beta;

	Householder_vector<RealD>(ck, 0, 2, v, beta);
	Householder_mult<RealD>(H,v,beta,0,lock_num+0,lock_num+2,0);
	Householder_mult<RealD>(H,v,beta,0,lock_num+0,lock_num+2,1);
	///Accumulate eigenvector
	Householder_mult<RealD>(Q,v,beta,0,lock_num+0,lock_num+2,1);
	  
	int sw = 0;
	for(int k=lock_num+0;k<M-2;k++){

	  x = H[k+1][k]; 
	  y = H[k+2][k]; 
	  z = (RealD)0.0;
	  if(k+3 <= M-1){
	    z = H[k+3][k];
	  }else{
	    sw = 1; v[2] = 0.0;
	  }

	  ck[0] = x; ck[1] = y; ck[2] = z;

	  normalise(ck);

	  Householder_vector<RealD>(ck, 0, 2-sw, v, beta);
	  Householder_mult<RealD>(H,v, beta,0,k+1,k+3-sw,0);
	  Householder_mult<RealD>(H,v, beta,0,k+1,k+3-sw,1);
	  ///Accumulate eigenvector
	  Householder_mult<RealD>(Q,v, beta,0,k+1,k+3-sw,1);
	}
      }
    }

    void TestConv(DenseMatrix<RealD> & H,int SS, 
		  DenseVector<Field> &bq, Field &bf,
		  DenseVector<RealD> &tevals, DenseVector<DenseVector<RealD> > &tevecs, 
		  int lock, int converged)
    {
      std::cout<<GridLogMessage << "Converged " << converged << " so far." << std::endl;
      int lock_num = lock ? converged : 0;
      int M = Nm;

      ///Active Factorization
      DenseMatrix<RealD> AH; Resize(AH,SS - lock_num,SS - lock_num );

      AH = GetSubMtx(H,lock_num, SS, lock_num, SS);

      int NN=tevals.size();
      int AHsize=SS-lock_num;

      RealD small=1.0e-16;
      Wilkinson<RealD>(AH, tevals, tevecs, small);

#ifndef USE_LAPACK
      EigenSort(tevals, tevecs);
#endif

      RealD resid_nrm=  norm2(bf);

      if(!lock) converged = 0;
#if 0
      for(int i = SS - lock_num - 1; i >= SS - Nk && i >= 0; --i){

	RealD diff = 0;
	diff = abs( tevecs[i][Nm - 1 - lock_num] ) * resid_nrm;

	std::cout<<GridLogMessage << "residual estimate " << SS-1-i << " " << diff << " of (" << tevals[i] << ")" << std::endl;

	if(diff < converged) {

	  if(lock) {
	    
	    DenseMatrix<RealD> Q; Resize(Q,M,M);
	    bool herm = true; 

	    Lock(H, Q, tevals[i], converged, small, SS, herm);

	    times_real(bq, Q, bq.size());
	    bf = Q[M - 1][M - 1]* bf;
	    lock_num++;
	  }
	  converged++;
	  std::cout<<GridLogMessage << " converged on eval " << converged << " of " << Nk << std::endl;
	} else {
	  break;
	}
      }
#endif
      std::cout<<GridLogMessage << "Got " << converged << " so far " <<std::endl;	
    }

    ///Check
    void Check(DenseVector<RealD> &evals,
	       DenseVector<DenseVector<RealD> > &evecs) {

      DenseVector<RealD> goodval(this->get);

#ifndef USE_LAPACK
      EigenSort(evals,evecs);
#endif

      int NM = Nm;

      DenseVector< DenseVector<RealD> > V; Size(V,NM);
      DenseVector<RealD> QZ(NM*NM);

      for(int i = 0; i < NM; i++){
	for(int j = 0; j < NM; j++){
	  // evecs[i][j];
	}
      }
    }


/**
   There is some matrix Q such that for any vector y
   Q.e_1 = y and Q is unitary.
**/
  template<class T>
  static T orthQ(DenseMatrix<T> &Q, DenseVector<T> y){
    int N = y.size();	//Matrix Size
    Fill(Q,0.0);
    T tau;
    for(int i=0;i<N;i++){
      Q[i][0]=y[i];
    }
    T sig = conj(y[0])*y[0];
    T tau0 = abs(sqrt(sig));
    
    for(int j=1;j<N;j++){
      sig += conj(y[j])*y[j]; 
      tau = abs(sqrt(sig) ); 	

      if(abs(tau0) > 0.0){
	
	T gam = conj( (y[j]/tau)/tau0 );
	for(int k=0;k<=j-1;k++){  
	  Q[k][j]=-gam*y[k];
	}
	Q[j][j]=tau0/tau;
      } else {
	Q[j-1][j]=1.0;
      }
      tau0 = tau;
    }
    return tau;
  }

/**
	There is some matrix Q such that for any vector y
	Q.e_k = y and Q is unitary.
**/
  template< class T>
  static T orthU(DenseMatrix<T> &Q, DenseVector<T> y){
    T tau = orthQ(Q,y);
    SL(Q);
    return tau;
  }


/**
	Wind up with a matrix with the first con rows untouched

say con = 2
	Q is such that Qdag H Q has {x, x, val, 0, 0, 0, 0, ...} as 1st colum
	and the matrix is upper hessenberg
	and with f and Q appropriately modidied with Q is the arnoldi factorization

**/

template<class T>
static void Lock(DenseMatrix<T> &H, 	///Hess mtx	
		 DenseMatrix<T> &Q, 	///Lock Transform
		 T val, 		///value to be locked
		 int con, 	///number already locked
		 RealD small,
		 int dfg,
		 bool herm)
{	


  //ForceTridiagonal(H);

  int M = H.dim;
  DenseVector<T> vec; Resize(vec,M-con);

  DenseMatrix<T> AH; Resize(AH,M-con,M-con);
  AH = GetSubMtx(H,con, M, con, M);

  DenseMatrix<T> QQ; Resize(QQ,M-con,M-con);

  Unity(Q);   Unity(QQ);
  
  DenseVector<T> evals; Resize(evals,M-con);
  DenseMatrix<T> evecs; Resize(evecs,M-con,M-con);

  Wilkinson<T>(AH, evals, evecs, small);

  int k=0;
  RealD cold = abs( val - evals[k]); 
  for(int i=1;i<M-con;i++){
    RealD cnew = abs( val - evals[i]);
    if( cnew < cold ){k = i; cold = cnew;}
  }
  vec = evecs[k];

  ComplexD tau;
  orthQ(QQ,vec);
  //orthQM(QQ,AH,vec);

  AH = Hermitian(QQ)*AH;
  AH = AH*QQ;
	

  for(int i=con;i<M;i++){
    for(int j=con;j<M;j++){
      Q[i][j]=QQ[i-con][j-con];
      H[i][j]=AH[i-con][j-con];
    }
  }

  for(int j = M-1; j>con+2; j--){

    DenseMatrix<T> U; Resize(U,j-1-con,j-1-con);
    DenseVector<T> z; Resize(z,j-1-con); 
    T nm = norm(z); 
    for(int k = con+0;k<j-1;k++){
      z[k-con] = conj( H(j,k+1) );
    }
    normalise(z);

    RealD tmp = 0;
    for(int i=0;i<z.size()-1;i++){tmp = tmp + abs(z[i]);}

    if(tmp < small/( (RealD)z.size()-1.0) ){ continue;}	

    tau = orthU(U,z);

    DenseMatrix<T> Hb; Resize(Hb,j-1-con,M);	
	
    for(int a = 0;a<M;a++){
      for(int b = 0;b<j-1-con;b++){
	T sum = 0;
	for(int c = 0;c<j-1-con;c++){
	  sum += H[a][con+1+c]*U[c][b];
	}//sum += H(a,con+1+c)*U(c,b);}
	Hb[b][a] = sum;
      }
    }
	
    for(int k=con+1;k<j;k++){
      for(int l=0;l<M;l++){
	H[l][k] = Hb[k-1-con][l];
      }
    }//H(Hb[k-1-con][l] , l,k);}}

    DenseMatrix<T> Qb; Resize(Qb,M,M);	
	
    for(int a = 0;a<M;a++){
      for(int b = 0;b<j-1-con;b++){
	T sum = 0;
	for(int c = 0;c<j-1-con;c++){
	  sum += Q[a][con+1+c]*U[c][b];
	}//sum += Q(a,con+1+c)*U(c,b);}
	Qb[b][a] = sum;
      }
    }
	
    for(int k=con+1;k<j;k++){
      for(int l=0;l<M;l++){
	Q[l][k] = Qb[k-1-con][l];
      }
    }//Q(Qb[k-1-con][l] , l,k);}}

    DenseMatrix<T> Hc; Resize(Hc,M,M);	
	
    for(int a = 0;a<j-1-con;a++){
      for(int b = 0;b<M;b++){
	T sum = 0;
	for(int c = 0;c<j-1-con;c++){
	  sum += conj( U[c][a] )*H[con+1+c][b];
	}//sum += conj( U(c,a) )*H(con+1+c,b);}
	Hc[b][a] = sum;
      }
    }

    for(int k=0;k<M;k++){
      for(int l=con+1;l<j;l++){
	H[l][k] = Hc[k][l-1-con];
      }
    }//H(Hc[k][l-1-con] , l,k);}}

  }
}
#endif


 };

}
#endif

