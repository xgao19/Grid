/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./tests/Test_simd.cc

Copyright (C) 2015

Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: neo <cossu@post.kek.jp>

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
#include <Grid/Grid.h>

using namespace std;
using namespace Grid;
using namespace Grid::QCD;

class funcPlus {
public:
  funcPlus() {};
  template<class vec> void operator()(vec &rr,vec &i1,vec &i2) const { rr = i1+i2;}
  std::string name(void) const { return std::string("Plus"); }
};
class funcMinus {
public:
  funcMinus() {};
  template<class vec> void operator()(vec &rr,vec &i1,vec &i2) const { rr = i1-i2;}
  std::string name(void) const { return std::string("Minus"); }
};
class funcTimes {
public:
  funcTimes() {};
  template<class vec> void operator()(vec &rr,vec &i1,vec &i2) const { rr = i1*i2;}
  std::string name(void) const { return std::string("Times"); }
};
class funcDivide {
public:
  funcDivide() {};
  template<class vec> void operator()(vec &rr,vec &i1,vec &i2) const { rr = i1/i2;}
  std::string name(void) const { return std::string("Divide"); }
};
class funcConj {
public:
  funcConj() {};
  template<class vec> void operator()(vec &rr,vec &i1,vec &i2) const { rr = conjugate(i1);}
  std::string name(void) const { return std::string("Conj"); }
};
class funcAdj {
public:
  funcAdj() {};
  template<class vec> void operator()(vec &rr,vec &i1,vec &i2) const { rr = adj(i1);}
  std::string name(void) const { return std::string("Adj"); }
};
class funcImag {
public:
  funcImag() {};
  template<class vec> void operator()(vec &rr,vec &i1,vec &i2) const { rr = imag(i1);}
  std::string name(void) const { return std::string("imag"); }
};
class funcReal {
public:
  funcReal() {};
  template<class vec> void operator()(vec &rr,vec &i1,vec &i2) const { rr = real(i1);}
  std::string name(void) const { return std::string("real"); }
};

class funcTimesI {
public:
  funcTimesI() {};
  template<class vec> void operator()(vec &rr,vec &i1,vec &i2) const { rr = timesI(i1);}
  std::string name(void) const { return std::string("timesI"); }
};
class funcTimesMinusI {
public:
  funcTimesMinusI() {};
  template<class vec> void operator()(vec &rr,vec &i1,vec &i2) const { rr = timesMinusI(i1);}
  std::string name(void) const { return std::string("timesMinusI"); }
};
class funcInnerProduct {
public:
  funcInnerProduct() {};
  template<class vec> void operator()(vec &rr,vec &i1,vec &i2) const { rr = innerProduct(i1,i2);}
  std::string name(void) const { return std::string("innerProduct"); }
};

// FIXME still to test:
//
//  innerProduct,
//  norm2, 
//  Reduce,
//
//  mac,mult,sub,add, vone,vzero,vcomplex_i, =zero,
//  vset,vsplat,vstore,vstream,vload, scalar*vec, vec*scalar
//  unary -,
//  *= , -=, +=
//  outerproduct, 
//  zeroit
//  permute


class funcReduce {
public:
  funcReduce() {};
template<class reduce,class vec>    void vfunc(reduce &rr,vec &i1,vec &i2)   const { rr = Reduce(i1);}
template<class reduce,class scal>   void sfunc(reduce &rr,scal &i1,scal &i2) const { rr = i1;}
  std::string name(void) const { return std::string("Reduce"); }
};

template<class scal, class vec,class functor > 
void Tester(const functor &func)
{
  GridSerialRNG          sRNG;
  sRNG.SeedRandomDevice();
  
  int Nsimd = vec::Nsimd();

  std::vector<scal> input1(Nsimd);
  std::vector<scal> input2(Nsimd);
  std::vector<scal> result(Nsimd);
  std::vector<scal> reference(Nsimd);

  std::vector<vec,alignedAllocator<vec> > buf(3);
  vec & v_input1 = buf[0];
  vec & v_input2 = buf[1];
  vec & v_result = buf[2];


  for(int i=0;i<Nsimd;i++){
    random(sRNG,input1[i]);
    random(sRNG,input2[i]);
    random(sRNG,result[i]);
  }

  merge<vec,scal>(v_input1,input1);
  merge<vec,scal>(v_input2,input2);
  merge<vec,scal>(v_result,result);

  func(v_result,v_input1,v_input2);

  for(int i=0;i<Nsimd;i++) {
    func(reference[i],input1[i],input2[i]);
  }

  extract<vec,scal>(v_result,result);

  std::cout << GridLogMessage << " " << func.name() << std::endl;

  std::cout << GridLogDebug << v_input1 << std::endl;
  std::cout << GridLogDebug << v_input2 << std::endl;
  std::cout << GridLogDebug << v_result << std::endl;

  int ok=0;
  for(int i=0;i<Nsimd;i++){
    if ( abs(reference[i]-result[i])>1.0e-7){
      std::cout<<GridLogMessage<< "*****" << std::endl;
      std::cout<<GridLogMessage<< "["<<i<<"] "<< abs(reference[i]-result[i]) << " " <<reference[i]<< " " << result[i]<<std::endl;
      ok++;
    }
  }
  if ( ok==0 ) {
    std::cout<<GridLogMessage << " OK!" <<std::endl;
  }
  assert(ok==0);
}

template<class functor>
void IntTester(const functor &func)
{
  typedef Integer  scal;
  typedef vInteger vec;
  GridSerialRNG          sRNG;
  sRNG.SeedRandomDevice();

  int Nsimd = vec::Nsimd();

  std::vector<scal> input1(Nsimd);
  std::vector<scal> input2(Nsimd);
  std::vector<scal> result(Nsimd);
  std::vector<scal> reference(Nsimd);

  std::vector<vec,alignedAllocator<vec> > buf(3);
  vec & v_input1 = buf[0];
  vec & v_input2 = buf[1];
  vec & v_result = buf[2];


  for(int i=0;i<Nsimd;i++){
    input1[i] = (i + 1) * 30;
    input2[i] = (i + 1) * 20;
    result[i] = (i + 1) * 10;
  }

  merge<vec,scal>(v_input1,input1);
  merge<vec,scal>(v_input2,input2);
  merge<vec,scal>(v_result,result);

  func(v_result,v_input1,v_input2);

  for(int i=0;i<Nsimd;i++) {
    func(reference[i],input1[i],input2[i]);
  }

  extract<vec,scal>(v_result,result);

  std::cout << GridLogMessage << " " << func.name() << std::endl;

  std::cout << GridLogDebug << v_input1 << std::endl;
  std::cout << GridLogDebug << v_input2 << std::endl;
  std::cout << GridLogDebug << v_result << std::endl;

  int ok=0;
  for(int i=0;i<Nsimd;i++){
    if ( reference[i]-result[i] != 0){
      std::cout<<GridLogMessage<< "*****" << std::endl;
      std::cout<<GridLogMessage<< "["<<i<<"] "<< reference[i]-result[i] << " " <<reference[i]<< " " << result[i]<<std::endl;
      ok++;
    }
  }
  if ( ok==0 ) {
    std::cout<<GridLogMessage << " OK!" <<std::endl;
  }
  assert(ok==0);
}


template<class reduced,class scal, class vec,class functor > 
void ReductionTester(const functor &func)
{
  GridSerialRNG          sRNG;
  sRNG.SeedRandomDevice();
  
  int Nsimd = vec::Nsimd();

  std::vector<scal> input1(Nsimd);
  std::vector<scal> input2(Nsimd);
  reduced result(0);
  reduced reference(0);
  reduced tmp;

  std::vector<vec,alignedAllocator<vec> > buf(3);
  vec & v_input1 = buf[0];
  vec & v_input2 = buf[1];


  for(int i=0;i<Nsimd;i++){
    random(sRNG,input1[i]);
    random(sRNG,input2[i]);
  }

  merge<vec,scal>(v_input1,input1);
  merge<vec,scal>(v_input2,input2);

  func.template vfunc<reduced,vec>(result,v_input1,v_input2);

  for(int i=0;i<Nsimd;i++) {
    func.template sfunc<reduced,scal>(tmp,input1[i],input2[i]);
    reference+=tmp;
  }

  std::cout<<GridLogMessage << " " << func.name()<<std::endl;

  int ok=0;
  if ( abs(reference-result)/abs(reference) > 1.0e-6 ){ // rounding is possible for reduce order
    std::cout<<GridLogMessage<< "*****" << std::endl;
    std::cout<<GridLogMessage<< abs(reference-result) << " " <<reference<< " " << result<<std::endl;
    ok++;
  }
  if ( ok==0 ) {
    std::cout<<GridLogMessage << " OK!" <<std::endl;
  }
  assert(ok==0);
}



class funcPermute {
public:
  int n;
  funcPermute(int _n) { n=_n;};
  template<class vec>    void operator()(vec &rr,vec &i1,vec &i2) const { permute(rr,i1,n);}
  template<class scal>   void apply(std::vector<scal> &rr,std::vector<scal> &in)  const { 
    int sz=in.size();
    int msk = sz>>(n+1);
    for(int i=0;i<sz;i++){
      rr[i] = in[ i^msk ];
    }
  }
  std::string name(void) const { return std::string("Permute"); }
};
class funcRotate {
public:
  int n;
  funcRotate(int _n) { n=_n;};
  template<class vec>    void operator()(vec &rr,vec &i1,vec &i2) const { rr=rotate(i1,n);}
  template<class scal>   void apply(std::vector<scal> &rr,std::vector<scal> &in)  const { 
    int sz = in.size();
    for(int i=0;i<sz;i++){
      rr[i] = in[(i+n)%sz];
    }
  }
  std::string name(void) const { return std::string("Rotate"); }
};


template<class scal, class vec,class functor > 
void PermTester(const functor &func)
{
  GridSerialRNG          sRNG;
  sRNG.SeedRandomDevice();
  
  int Nsimd = vec::Nsimd();

  std::vector<scal> input1(Nsimd);
  std::vector<scal> input2(Nsimd);
  std::vector<scal> result(Nsimd);
  std::vector<scal> reference(Nsimd);

  std::vector<vec,alignedAllocator<vec> > buf(3);
  vec & v_input1 = buf[0];
  vec & v_input2 = buf[1];
  vec & v_result = buf[2];

  for(int i=0;i<Nsimd;i++){
    random(sRNG,input1[i]);
    random(sRNG,input2[i]);
    random(sRNG,result[i]);
  }

  merge<vec,scal>(v_input1,input1);
  merge<vec,scal>(v_input2,input2);
  merge<vec,scal>(v_result,result);

  func(v_result,v_input1,v_input2);

  func.apply(reference,input1);

  extract<vec,scal>(v_result,result);
  std::cout<<GridLogMessage << " " << func.name() << " " <<func.n <<std::endl;

  int ok=0;
  if (0) {
    std::cout<<GridLogMessage<< "*****" << std::endl;
    for(int i=0;i<Nsimd;i++){
      std::cout<< input1[i]<<" ";
    }
    std::cout <<std::endl; 
    for(int i=0;i<Nsimd;i++){
      std::cout<< result[i]<<" ";
    }
    std::cout <<std::endl; 
    for(int i=0;i<Nsimd;i++){
      std::cout<< reference[i]<<" ";
    }
    std::cout <<std::endl; 
    std::cout<<GridLogMessage<< "*****" << std::endl;
  }
  for(int i=0;i<Nsimd;i++){
    if ( abs(reference[i]-result[i])>1.0e-7){
      std::cout<<GridLogMessage<< "*****" << std::endl;      
      std::cout<<GridLogMessage<< "["<<i<<"] "<< abs(reference[i]-result[i]) << " " <<reference[i]<< " " << result[i]<<std::endl;
      ok++;
    }
  }
  if ( ok==0 ) {
    std::cout<<GridLogMessage << " OK!" <<std::endl;
  }
  assert(ok==0);
}

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  std::vector<int> latt_size   = GridDefaultLatt();
  std::vector<int> simd_layout = GridDefaultSimd(4,vComplex::Nsimd());
  std::vector<int> mpi_layout  = GridDefaultMpi();
    
  GridCartesian     Grid(latt_size,simd_layout,mpi_layout);
  std::vector<int> seeds({1,2,3,4});

  // Insist that operations on random scalars gives
  // identical results to on vectors.

  std::cout << GridLogMessage <<"==================================="<<  std::endl;
  std::cout << GridLogMessage <<"Testing vRealF "<<std::endl;
  std::cout << GridLogMessage <<"==================================="<<  std::endl;


  Tester<RealF,vRealF>(funcPlus());
  Tester<RealF,vRealF>(funcMinus());
  Tester<RealF,vRealF>(funcTimes());
  Tester<RealF,vRealF>(funcDivide());
  Tester<RealF,vRealF>(funcAdj());
  Tester<RealF,vRealF>(funcConj());
  Tester<RealF,vRealF>(funcInnerProduct());
  ReductionTester<RealF,RealF,vRealF>(funcReduce());


  std::cout<<GridLogMessage << "==================================="<<  std::endl;
  std::cout<<GridLogMessage << "Testing vRealF permutes "<<std::endl;
  std::cout<<GridLogMessage << "==================================="<<  std::endl;

  // Log2 iteration
  for(int i=0;(1<<i)< vRealF::Nsimd();i++){
    PermTester<RealF,vRealF>(funcPermute(i));
  }

  std::cout<<GridLogMessage << "==================================="<<  std::endl;
  std::cout<<GridLogMessage << "Testing vRealF rotate "<<std::endl;
  std::cout<<GridLogMessage << "==================================="<<  std::endl;
  for(int r=0;r<vRealF::Nsimd();r++){
    PermTester<RealF,vRealF>(funcRotate(r));
  }


  std::cout << GridLogMessage <<"==================================="<<  std::endl;
  std::cout << GridLogMessage <<"Testing vRealD "<<std::endl;
  std::cout << GridLogMessage <<"==================================="<<  std::endl;

  Tester<RealD,vRealD>(funcPlus());
  Tester<RealD,vRealD>(funcMinus());
  Tester<RealD,vRealD>(funcTimes());
  Tester<RealD,vRealD>(funcDivide());
  Tester<RealD,vRealD>(funcAdj());
  Tester<RealD,vRealD>(funcConj());
  Tester<RealD,vRealD>(funcInnerProduct());
  ReductionTester<RealD,RealD,vRealD>(funcReduce());


  std::cout<<GridLogMessage << "==================================="<<  std::endl;
  std::cout<<GridLogMessage << "Testing vRealD permutes "<<std::endl;
  std::cout<<GridLogMessage << "==================================="<<  std::endl;

  // Log2 iteration
  for(int i=0;(1<<i)< vRealD::Nsimd();i++){
    PermTester<RealD,vRealD>(funcPermute(i));
  }

  std::cout<<GridLogMessage << "==================================="<<  std::endl;
  std::cout<<GridLogMessage << "Testing vRealD rotate "<<std::endl;
  std::cout<<GridLogMessage << "==================================="<<  std::endl;
  for(int r=0;r<vRealD::Nsimd();r++){
    PermTester<RealD,vRealD>(funcRotate(r));
  }



  std::cout << GridLogMessage <<"==================================="<<  std::endl;
  std::cout << GridLogMessage <<"Testing vComplexF "<<std::endl;
  std::cout << GridLogMessage <<"==================================="<<  std::endl;

  Tester<ComplexF,vComplexF>(funcTimesI());
  Tester<ComplexF,vComplexF>(funcTimesMinusI());
  Tester<ComplexF,vComplexF>(funcPlus());
  Tester<ComplexF,vComplexF>(funcMinus());
  Tester<ComplexF,vComplexF>(funcTimes());
  Tester<ComplexF,vComplexF>(funcConj());
  Tester<ComplexF,vComplexF>(funcAdj());
  Tester<ComplexF,vComplexF>(funcReal());
  Tester<ComplexF,vComplexF>(funcImag());
  Tester<ComplexF,vComplexF>(funcInnerProduct());
  ReductionTester<ComplexF,ComplexF,vComplexF>(funcReduce());


  std::cout<<GridLogMessage << "==================================="<<  std::endl;
  std::cout<<GridLogMessage << "Testing vComplexF permutes "<<std::endl;
  std::cout<<GridLogMessage << "==================================="<<  std::endl;

  // Log2 iteration
  for(int i=0;(1<<i)< vComplexF::Nsimd();i++){
    PermTester<ComplexF,vComplexF>(funcPermute(i));
  }

  std::cout<<GridLogMessage << "==================================="<<  std::endl;
  std::cout<<GridLogMessage << "Testing vComplexF rotate "<<std::endl;
  std::cout<<GridLogMessage << "==================================="<<  std::endl;
  for(int r=0;r<vComplexF::Nsimd();r++){
    PermTester<ComplexF,vComplexF>(funcRotate(r));
  }

  std::cout<<GridLogMessage << "==================================="<<  std::endl;
  std::cout<<GridLogMessage << "Testing vComplexD "<<std::endl;
  std::cout<<GridLogMessage << "==================================="<<  std::endl;


  Tester<ComplexD,vComplexD>(funcTimesI());
  Tester<ComplexD,vComplexD>(funcTimesMinusI());
  Tester<ComplexD,vComplexD>(funcPlus());
  Tester<ComplexD,vComplexD>(funcMinus());
  Tester<ComplexD,vComplexD>(funcTimes());
  Tester<ComplexD,vComplexD>(funcConj());
  Tester<ComplexD,vComplexD>(funcAdj());
  Tester<ComplexD, vComplexD>(funcReal());
  Tester<ComplexD, vComplexD>(funcImag());

  Tester<ComplexD, vComplexD>(funcInnerProduct());
  ReductionTester<ComplexD, ComplexD, vComplexD>(funcReduce());

  std::cout << GridLogMessage
            << "===================================" << std::endl;
  std::cout << GridLogMessage << "Testing vComplexD permutes " << std::endl;
  std::cout << GridLogMessage
            << "===================================" << std::endl;

  // Log2 iteration
  for (int i = 0; (1 << i) < vComplexD::Nsimd(); i++) {
    PermTester<ComplexD, vComplexD>(funcPermute(i));
  }


  std::cout<<GridLogMessage << "==================================="<<  std::endl;
  std::cout<<GridLogMessage << "Testing vComplexD rotate "<<std::endl;
  std::cout<<GridLogMessage << "==================================="<<  std::endl;
  for(int r=0;r<vComplexD::Nsimd();r++){
    PermTester<ComplexD,vComplexD>(funcRotate(r));
  }
  
  std::cout<<GridLogMessage << "==================================="<<  std::endl;
  std::cout<<GridLogMessage << "Testing vInteger                   "<<  std::endl;
  std::cout<<GridLogMessage << "==================================="<<  std::endl;
  IntTester(funcPlus());
  IntTester(funcMinus());
  IntTester(funcTimes());

  Grid_finalize();
}
