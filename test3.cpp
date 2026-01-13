// ReSharper disable CppTemplateArgumentsCanBeDeduced
#include <petscksp.h>
#include <vector>

PetscErrorCode MatSetPreallocationCOO_Matrix(std::vector<PetscInt> A_col, std::vector<PetscInt> A_row, Mat A){
    const uint nnz_col = A_col.size();
    const uint nnz_row = A_row.size();
    PetscFunctionBeginUser;
    PetscCheck(nnz_col == nnz_row, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, nullptr);
    PetscCall(MatSetPreallocationCOO(A, nnz_col, A_row.data(), A_col.data()));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode MatSetLocal_Matrix(const uint nnz, const std::vector<PetscScalar>& A_val, Mat A){
    PetscFunctionBeginUser;
    PetscCheck(A_val.size() == nnz, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, nullptr);
    PetscCall(MatSetValuesCOO(A, A_val.data(), INSERT_VALUES));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode Finalize(){
    PetscBool init = PETSC_FALSE;
    PetscFunctionBeginUser;
    PetscCall(PetscInitialized(&init));
    if(init){
        PetscCall(PetscFinalize());
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char** argv){
#ifdef WITH_SLEPC
    PetscCall(SlepcInitialize(&argc, &argv, nullptr, nullptr));
#else
    PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
#endif
    PetscCall(PetscLogDefaultBegin());

    // EXAMPLE 1: A = [ 0,  2,  0,  1,  0;
    //                  4, -1, -1,  0,  0;
    //                  0,  0,  0,  3, -6;
    //                 -2,  0,  0,  0,  2;
    //                  0,  0,  4,  2,  0];
    PetscInt                 m     = 5, n = m;
    PetscInt                 nnz   = 11;
    std::vector<PetscInt>    A_row = {0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4};
    std::vector<PetscInt>    A_col = {1, 3, 0, 1, 2, 3, 4, 0, 4, 2, 3};
    std::vector<PetscScalar> A_val = {2, 1, 4, -1, -1, 3, -6, -2, 2, 4, 2};
    std::vector<PetscInt>    b_row = {0, 1, 2, 3, 4};
    std::vector<PetscScalar> b_val = {8, -1, -18, 8, 20};
    PetscInt                 M = PETSC_DECIDE, N = M;
    Mat                      A;
    Vec                      b, x;
    KSP                      ksp;
    PC                       pc;

    // create
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(PetscSplitOwnership(PETSC_COMM_WORLD, &m, &M));
    PetscCall(PetscSplitOwnership(PETSC_COMM_WORLD, &n, &N));
    PetscCall(MatSetSizes(A, m, n, M, N));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetUp(A));

    // mat
    PetscCall(MatSetPreallocationCOO_Matrix(A_col, A_row, A));
    PetscCall(MatSetLocal_Matrix(nnz, A_val, A));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    // vec
    PetscCall(MatCreateVecs(A, &b, &x));
    PetscCall(VecSetValues(b, 5, b_row.data(), b_val.data(), INSERT_VALUES));
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));

    // aux. calculation
    /*
    PetscCall(MatMult(A, b, x));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "b=\n"));
    PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));
    */

    // solve for x
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCFactorSetReuseOrdering(pc, PETSC_TRUE));
    PetscCall(PCFactorSetReuseFill(pc, PETSC_TRUE));
    PetscCall(PCSetFromOptions(pc));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(MatViewFromOptions(A, nullptr, "-A_view"));
    PetscCall(VecViewFromOptions(b, nullptr, "-b_view"));
    PetscCall(VecViewFromOptions(x, nullptr, "-x_view"));

    // destroy
    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&x));

    // finalise
    PetscCall(Finalize());

    PetscFunctionReturn(PETSC_SUCCESS);
}

// -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -A_view -b_view -x_view -malloc_dump -malloc_debug -malloc_dump
// -ksp_type gmres -pc_type jacobi -A_view -b_view -x_view -malloc_dump -malloc_debug -malloc_dump
// https://github.com/FilipeFT24/PETSc-repo/blob/master/ex81a.cpp
// https://gitlab.com/petsc/petsc/-/blob/fccfa4b43f74198ab890b077de64eee091c20710/src/ksp/ksp/tutorials/ex85.c
// https://gitlab.inria.fr/aerosol/aerosol/-/blob/fft_phd/lib/include/common/SMatrixPETSc.hpp?ref_type=heads