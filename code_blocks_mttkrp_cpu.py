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

IndexStmt scheduleSpMMCPU_test(IndexStmt stmt, Tensor<double> B, IndexExpr precomputedExpr, int num, TensorVar precomputed) {
    IndexVar  x0("x0"), x1("x1"), x2("x2"), x3("x3"), x4("x4"), x5("x5"), x6("x6"),
    x7("x7"), x8("x8"), x9("x9"), x10("x10"), x11("x11");
    IndexVar fused_split_var("fsv"), f0("f0"), f1("f1");
    IndexVar vec_split_var("vsv"), vec_outer("vo"), vec_inner("v1");
    IndexVar block("block"), warp("warp"), thread("thread");
    IndexVar thread_unbounded("thread_unbounded"), thread_nz("thread_nz"), thread_nz_unbounded("thread_nz_unbounded")
    , thread_nz_bounded("thread_nz_bounded");
    //TensorVar precomputed("precomputed", Type(Float64, {Dimension(j)}), taco::dense);

"""



END="""
}

int main(int argc, char* argv[]) {

 int NUM_I = 1021/40;
  int NUM_J = 32;
  int NUM_K = 1039/40;
  int NUM_L = 1232/40;
  float SPARSITY = .1;
  Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<double> B("B", {NUM_I, NUM_K, NUM_L}, {Dense, Sparse, Sparse});
  Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
  Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});

  srand(5464164);
  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      for (int l = 0; l < NUM_L; l++) {
        float rand_float = (float) rand() / (float) (RAND_MAX);
        if (rand_float < SPARSITY) {
          B.insert({i, k, l}, (double) ((int) (rand_float * 3 / SPARSITY)));
        }
      }
    }
  }

  for (int k = 0; k < NUM_K; k++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({k, j}, (double) ((int) (rand_float*3)));
    }
  }

  for (int l = 0; l < NUM_L; l++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({l, j}, (double) ((int) (rand_float*3)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

    IndexExpr precomputedExpr = B(i,k,l) * D(l,j);
    A(i,j) = precomputedExpr * C(k,j);
  
  

  Tensor<double> expected("expected",{NUM_I,NUM_J},{Dense,Dense});
  expected(i,j) = B(i,k,l) * C(k,j) * D(l,j);
  expected.compile();
  expected.assemble();
  expected.compute();
  double * expected_ptr = (double *) expected.getStorage().getValues().getData();

 for(int num =0; num < NUMSCHEDULES; num ++)
 {
 IndexStmt stmt;
  std::cout << "Attempting to compile schedule " + to_string(num) << std::endl;
 TensorVar precomputed("precomputed", Type(Float64, {Dimension(j)}), taco::dense);

  IndexStmt precomputed_stmt = forall(i, forall(k,
                      where(forall(j, A(i,j) += precomputed(j) * C(k,j)),
                            forall(l, forall(j, precomputed(j) += B(i,k,l) * D(l,j))))));

  try{
         stmt = scheduleSpMMCPU_test(precomputed_stmt,B,precomputedExpr, num, precomputed);
          //A.compile(stmt);
          //A.assemble();
          //A.compute();
          printToFile("BASE_" + to_string(num), stmt);
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
  
  
  
 }
}
"""

