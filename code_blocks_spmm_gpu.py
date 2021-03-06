BEGIN = """
DEFINEBLOCK

#include <iostream>
#include <chrono>
#include <taco/index_notation/transformations.h>
#include <../src/codegen/codegen_c.h>
#include <../src/codegen/codegen_cuda.h>
#include <fstream>
#include "../test/test.h"
#include "../test/test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "../src/codegen/codegen.h"
#include "taco/lower/lower.h"

using namespace taco;

void printToFile(string filename, IndexStmt stmt) {
  stringstream source;

  string file_path = "eval_generated/";
  mkdir(file_path.c_str(), 0777);

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  codegen->compile(compute, true);

  ofstream source_file;
  string file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";
  source_file.open(file_path + filename + file_ending);
  source_file << source.str();
  source_file.close();
}


const IndexVar i("i"), j("j"), k("k"), l("l");

IndexStmt scheduleSpMMCPU_test(IndexStmt stmt, Tensor<double> A, int num) {
    IndexVar  x0("x0"), x1("x1"), x2("x2"), x3("x3"), x4("x4"), x5("x5"), x6("x6"),
    x7("x7"), x8("x8"), x9("x9"), x10("x10"), x11("x11");
    IndexVar fused_split_var("fsv"), f0("f0"), f1("f1");
    IndexVar vec_split_var("vsv"), vec_outer("vo"), vec_inner("v1");
    IndexVar block("block"), warp("warp"), thread("thread");
    IndexVar thread_unbounded("thread_unbounded"), thread_nz("thread_nz"), thread_nz_unbounded("thread_nz_unbounded")
    , thread_nz_bounded("thread_nz_bounded");

"""



END="""
}

int main(int argc, char* argv[]) {

  int NUM_I = 1024 ;
  int NUM_J = 1024;
  int NUM_K = 128;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> B("B", {NUM_J, NUM_K}, {Dense,Dense});
  Tensor<double> C("C", {NUM_K, NUM_I}, {Dense,Dense});
  //std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);

  srand(4353);
  for (int ii = 0; ii < NUM_I; ii++) {
    for (int jj = 0; jj < NUM_J; jj++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({ii, jj}, (double)((int) (rand_float*3/SPARSITY)));
      }
    }
  }
  for (int jj = 0; jj < NUM_J; jj++) {
    for (int kk = 0; kk < NUM_K; kk++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({jj, kk}, (double)((int) (rand_float*3/SPARSITY)));
      }
    }
  }
  A.pack();
  B.pack();
  C.pack();

  Tensor<double> expected("expected",{NUM_K,NUM_I},{Dense,Dense});
  expected(k,i) = A(i,j) * B(j,k);
  expected.compile();
  expected.assemble();
  expected.compute();
  double * expected_ptr = (double *) expected.getStorage().getValues().getData();

 for(int num =0; num < NUMSCHEDULES; num ++)
 {
 
  std::cout << "Attempting to compile schedule " + to_string(num) << std::endl;
 
  C(k,i) = A(i,j) * B(j,k);
  IndexStmt stmt = C.getAssignment().concretize();

  try{
        stmt = scheduleSpMMCPU_test(stmt,A,num);
          C.compile(stmt);
          C.assemble();
         //  C.compute();
  }
  catch (int e)
  {
        if(e == 2)
        {
            std::cout << "bad parameter" << std::endl;
        }
        std::cout << "cannot compile" << std::endl;
        continue;
  }
  printToFile("BASE_" + to_string(num), stmt);
  /*
  double * C_ptr = (double *) C.getStorage().getValues().getData();


  bool flag = false;
  for(int i = 0; i < NUM_I * NUM_K; i ++)
  {
          if(C_ptr[i] != expected_ptr[i])
          {
                  std::cout << i << std::endl;
                  flag = true;
                  break;

          }
  }
  if(flag)
  {
          std::cout << "wrong result" << std::endl;
          continue;
  }


  printToFile("BASE_" + to_string(num), stmt);*/
 }
}
"""

