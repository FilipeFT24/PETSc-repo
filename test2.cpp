// ReSharper disable CppTemplateArgumentsCanBeDeduced
#include <petscksp.h>
#include <vector>

PetscErrorCode MatSetPreallocationCOO_SubMatrix(std::vector<PetscInt> Ai_col, std::vector<PetscInt> Ai_row, Mat Ai){
    const uint nnz_col = Ai_col.size();
    const uint nnz_row = Ai_row.size();
    PetscFunctionBeginUser;
    PetscCheck(nnz_col == nnz_row, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, nullptr);
    PetscCall(MatSetPreallocationCOO(Ai, nnz_col, Ai_row.data(), Ai_col.data()));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode MatSetLocal_SubMatrix(const uint nnz, const PetscInt ib, const PetscInt jb, const std::vector<PetscScalar>& Ai_val, Mat A){
    Mat Ai;
    PetscFunctionBeginUser;
    PetscCall(MatNestGetSubMat(A, ib, jb, &Ai));
    PetscCheck(Ai_val.size() == nnz, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, nullptr);
    PetscCall(MatSetValuesCOO(Ai, Ai_val.data(), INSERT_VALUES));
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

    // EXAMPLE 1: A = [A0, A1, A2; A3, A4, A5; A6, A7, A8], with A0 = [0, 2; 4, -1], A1 = [0; -1], A2 = [1, 0; 0, 1]
    //                                                           A3 = [0, 0], A4 = [0], A5 = [3, -6]
    //                                                           A6 = [-2, 0; 0, 0], A7 = [0; 4], A8 = [0, 2; 2, 0]
    //            b = [b0; b1; b2], with                         b0 = [8; -1]
    //                                                           b1 = [-18]
    //                                                           b2 = [8; 20]
    PetscInt                 m        = 5, n = m;
    PetscInt                 D        = 3;
    PetscInt                 DD       = D*D;
    PetscInt                 di_row[] = {2, 2, 2, 1, 1, 1, 2, 2, 2};
    PetscInt                 di_col[] = {2, 1, 2, 2, 1, 2, 2, 1, 2};
    PetscInt                 nnz   [] = {4, 2, 4, 2, 1, 2, 4, 2, 4};
    std::vector<PetscInt>    A0_row   = {0, 0, 1, 1};
    std::vector<PetscInt>    A0_col   = {0, 1, 0, 1};
    std::vector<PetscInt>    A1_row   = {0, 1};
    std::vector<PetscInt>    A1_col   = {0, 0};
    std::vector<PetscInt>    A2_row   = {0, 0, 1, 1};
    std::vector<PetscInt>    A2_col   = {0, 1, 0, 1};
    std::vector<PetscInt>    A3_row   = {0, 0};
    std::vector<PetscInt>    A3_col   = {0, 1};
    std::vector<PetscInt>    A4_row   = {0};
    std::vector<PetscInt>    A4_col   = {0};
    std::vector<PetscInt>    A5_row   = {0, 0};
    std::vector<PetscInt>    A5_col   = {0, 1};
    std::vector<PetscInt>    A6_row   = {0, 0, 1, 1};
    std::vector<PetscInt>    A6_col   = {0, 1, 0, 1};
    std::vector<PetscInt>    A7_row   = {0, 1};
    std::vector<PetscInt>    A7_col   = {0, 0};
    std::vector<PetscInt>    A8_row   = {0, 0, 1, 1};
    std::vector<PetscInt>    A8_col   = {0, 1, 0, 1};
    std::vector<PetscScalar> A0_val   = {0, 2, 4, -1};
    std::vector<PetscScalar> A1_val   = {0, -1};
    std::vector<PetscScalar> A2_val   = {1, 0, 0, 0};
    std::vector<PetscScalar> A3_val   = {0, 0};
    std::vector<PetscScalar> A4_val   = {0};
    std::vector<PetscScalar> A5_val   = {3, -6};
    std::vector<PetscScalar> A6_val   = {-2, 0, 0, 0};
    std::vector<PetscScalar> A7_val   = {0, 4};
    std::vector<PetscScalar> A8_val   = {0, 2, 2, 0};
    std::vector<PetscInt>    b0_row   = {0, 1};
    std::vector<PetscInt>    b1_row   = {0};
    std::vector<PetscInt>    b2_row   = {0, 1};
    std::vector<PetscScalar> b0_val   = {8, -1};
    std::vector<PetscScalar> b1_val   = {-18};
    std::vector<PetscScalar> b2_val   = {8, 20};

    PetscInt                 M = PETSC_DECIDE, N = M;
    PetscInt                 rs, cs;
    PetscInt                 rsA;
    PetscInt*                loccA;
    PetscInt*                locrA;
    ISLocalToGlobalMapping   rowmA, colmA;
    Mat                      A;
    IS                       rows[D];
    Mat*                     Ai;
    PetscLayout              layoutA;
    Vec                      b, x;
    Vec                      bi;
    KSP                      ksp;
    PC                       pc;

    // create
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetType(A, MATNEST));
    PetscCall(PetscSplitOwnership(PETSC_COMM_WORLD, &m, &M));
    PetscCall(PetscSplitOwnership(PETSC_COMM_WORLD, &n, &N));
    PetscCall(MatSetSizes(A, m, n, M, N));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetUp(A));
    PetscCall(MatSetOption(A, MAT_ROW_ORIENTED, PETSC_TRUE)); // -1
    PetscCall(MatSetOption(A, MAT_STRUCTURALLY_SYMMETRIC, PETSC_TRUE)); // 2
    PetscCall(MatSetOption(A, MAT_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE)); // 4
    PetscCall(MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE)); // 6
    PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE)); // 11
    PetscCall(MatSetOption(A, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE)); // 16
    PetscCall(MatSetOption(A, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE)); // 17
    PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE)); // 19
    PetscCall(MatSetOption(A, MAT_SORTED_FULL, PETSC_TRUE)); // 23
    PetscCall(MatGetOwnershipRange(A, &rs, nullptr));
    PetscCall(MatGetOwnershipRangeColumn(A, &cs, nullptr));

    // layout
    for(PetscInt i = 0; i < D; ++i){
        if(i > 0 && i < D-1) {
            continue;
        }
        PetscInt j = i*(D+1);
        PetscCall(PetscLayoutCreate(PETSC_COMM_WORLD, &layoutA));
        PetscCall(PetscLayoutSetLocalSize(layoutA, di_row[j]));
        PetscCall(PetscLayoutSetUp(layoutA));
        PetscCall(PetscLayoutGetRange(layoutA, &rsA, nullptr));
        PetscCall(PetscLayoutDestroy(&layoutA));
        PetscCall(PetscMalloc1(di_row[j], &locrA));
        PetscCall(PetscMalloc1(di_col[j], &loccA));
        for(PetscInt r = 0; r < di_row[j]; ++r){
            locrA[r] = r+rsA;
        }
        for(PetscInt c = 0; c < di_col[j]; ++c){
            loccA[c] = c+rsA;
        }
        PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, di_row[j], locrA, PETSC_OWN_POINTER, &rowmA));
        PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, di_col[j], loccA, PETSC_OWN_POINTER, &colmA));
        PetscCall(ISLocalToGlobalMappingDestroy(&rowmA));
        PetscCall(ISLocalToGlobalMappingDestroy(&colmA));
    }

    // submat
    PetscCall(PetscCalloc1(DD, &Ai));
    /*
    for(PetscInt i = 0; i < D; ++i){
        PetscCall(MatCreate(PETSC_COMM_WORLD, &Ai[i*(D+1)]));
        PetscCall(MatSetSizes(Ai[i*(D+1)], di_row[i], di_col[i], PETSC_DECIDE, PETSC_DECIDE));
    }
    for(PetscInt i = 0; i < D; ++i){
        PetscCall(MatSetUp(Ai[i*(D+1)]));
    }
    */
    for(PetscInt i = 0; i < DD; ++i){
        PetscCall(MatCreate(PETSC_COMM_WORLD, &Ai[i]));
        PetscCall(MatSetSizes(Ai[i], di_row[i], di_col[i], PETSC_DECIDE, PETSC_DECIDE));
    }
    for(PetscInt i = 0; i < DD; ++i){
        PetscCall(MatSetUp(Ai[i]));
    }
    PetscCall(MatSetPreallocationCOO_SubMatrix(A0_col, A0_row, Ai[0]));
    PetscCall(MatSetPreallocationCOO_SubMatrix(A1_col, A1_row, Ai[1]));
    PetscCall(MatSetPreallocationCOO_SubMatrix(A2_col, A2_row, Ai[2]));
    PetscCall(MatSetPreallocationCOO_SubMatrix(A3_col, A3_row, Ai[3]));
    PetscCall(MatSetPreallocationCOO_SubMatrix(A4_col, A4_row, Ai[4]));
    PetscCall(MatSetPreallocationCOO_SubMatrix(A5_col, A5_row, Ai[5]));
    PetscCall(MatSetPreallocationCOO_SubMatrix(A6_col, A6_row, Ai[6]));
    PetscCall(MatSetPreallocationCOO_SubMatrix(A7_col, A7_row, Ai[7]));
    PetscCall(MatSetPreallocationCOO_SubMatrix(A8_col, A8_row, Ai[8]));

    // mat
    PetscCall(MatNestSetSubMats(A, D, nullptr, D, nullptr, Ai));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    for(PetscInt i = 0; i < DD; ++i){
        PetscCall(MatDestroy(&Ai[i]));
    }
    PetscCall(PetscFree(Ai));

    // submat
    PetscCall(MatSetLocal_SubMatrix(nnz[0], 0, 0, A0_val, A));
    PetscCall(MatSetLocal_SubMatrix(nnz[1], 0, 1, A1_val, A));
    PetscCall(MatSetLocal_SubMatrix(nnz[2], 0, 2, A2_val, A));
    PetscCall(MatSetLocal_SubMatrix(nnz[3], 1, 0, A3_val, A));
    PetscCall(MatSetLocal_SubMatrix(nnz[4], 1, 1, A4_val, A));
    PetscCall(MatSetLocal_SubMatrix(nnz[5], 1, 2, A5_val, A));
    PetscCall(MatSetLocal_SubMatrix(nnz[6], 2, 0, A6_val, A));
    PetscCall(MatSetLocal_SubMatrix(nnz[7], 2, 1, A7_val, A));
    PetscCall(MatSetLocal_SubMatrix(nnz[8], 2, 2, A8_val, A));

    // vec
    PetscCall(MatCreateVecs(A, &b, &x));
    PetscCall(MatNestGetISs(A, rows, nullptr));
    PetscCall(VecGetSubVector(b, rows[0], &bi));
    PetscCall(VecSetValues(bi, 2, b0_row.data(), b0_val.data(), INSERT_VALUES));
    PetscCall(VecRestoreSubVector(b, rows[0], &bi));
    PetscCall(VecGetSubVector(b, rows[1], &bi));
    PetscCall(VecSetValues(bi, 1, b1_row.data(), b1_val.data(), INSERT_VALUES));
    PetscCall(VecRestoreSubVector(b, rows[1], &bi));
    PetscCall(VecGetSubVector(b, rows[2], &bi));
    PetscCall(VecSetValues(bi, 2, b2_row.data(), b2_val.data(), INSERT_VALUES));
    PetscCall(VecRestoreSubVector(b, rows[2], &bi));

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

// -ksp_type preonly -pc_type lu -A_view -b_view -x_view -mat_view_nest_sub -malloc_dump -malloc_debug -malloc_dump // REVIEW
// https://github.com/FilipeFT24/PETSc-repo/blob/master/ex81a.cpp
// https://gitlab.com/petsc/petsc/-/blob/fccfa4b43f74198ab890b077de64eee091c20710/src/ksp/ksp/tutorials/ex85.c
// https://gitlab.inria.fr/aerosol/aerosol/-/blob/fft_phd/lib/include/common/SMatrixPETSc.hpp?ref_type=heads