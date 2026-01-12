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
PetscErrorCode MatSetLocal_SubMatrix(const PetscInt ib, const PetscInt jb, const PetscScalar* Ai_val, Mat A){
    Mat Ai;
    PetscFunctionBeginUser;
    PetscCall(MatNestGetSubMat(A, ib, jb, &Ai));
    PetscCall(MatSetValuesCOO(Ai, Ai_val, INSERT_VALUES));
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

    PetscInt    m        = 9, n = m, M = PETSC_DECIDE, N = PETSC_DECIDE, rStart, cStart;
    PetscInt    D        = 3;
    PetscInt    DD       = D*D;
    PetscBool   is_nest  = PETSC_FALSE;
    PetscInt    di_row[] = {3, 2, 4};
    PetscInt    di_col[] = {3, 2, 4};
    std::vector A1_row   = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    std::vector A1_col   = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector A2_row   = {0, 0, 1, 1};
    std::vector A2_col   = {0, 1, 0, 1};
    std::vector A3_row   = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    std::vector A3_col   = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    PetscScalar A1_val[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    PetscScalar A2_val[] = {9, 10, 11, 12};
    PetscScalar A3_val[] = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
    Mat         A;

    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
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
    PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(A), MATNEST, &is_nest));
    PetscCall(MatGetOwnershipRange(A, &rStart, nullptr));
    PetscCall(MatGetOwnershipRangeColumn(A, &cStart, nullptr));


    if(is_nest){
        Mat* Ai;
        PetscLayout layoutA1, layoutA2;
        PetscInt *locrA1, *loccA1, *locrA2, *loccA2, rsA1, rsA2;
        ISLocalToGlobalMapping rowmA1, colmA1, rowmA2, colmA2;

        // submat (A1)
        PetscCall(PetscLayoutCreate(PETSC_COMM_WORLD, &layoutA1));
        PetscCall(PetscLayoutSetLocalSize(layoutA1, di_row[0]));
        PetscCall(PetscLayoutSetUp(layoutA1));
        PetscCall(PetscLayoutGetRange(layoutA1, &rsA1, nullptr));
        PetscCall(PetscLayoutDestroy(&layoutA1));
        PetscCall(PetscMalloc1(di_row[0], &locrA1));
        PetscCall(PetscMalloc1(di_col[0], &loccA1));
        for(PetscInt r = 0; r < di_row[0]; ++r){
            locrA1[r] = r+rsA1;
        }
        for(PetscInt c = 0; c < di_col[0]; ++c){
            loccA1[c] = c+rsA1;
        }
        PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, di_row[0], locrA1, PETSC_OWN_POINTER, &rowmA1));
        PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, di_col[0], loccA1, PETSC_OWN_POINTER, &colmA1));
        PetscCall(ISLocalToGlobalMappingDestroy(&rowmA1));
        PetscCall(ISLocalToGlobalMappingDestroy(&colmA1));
        // submat (A2)
        PetscCall(PetscLayoutCreate(PETSC_COMM_WORLD, &layoutA2));
        PetscCall(PetscLayoutSetLocalSize(layoutA2, di_row[2]));
        PetscCall(PetscLayoutSetUp(layoutA2));
        PetscCall(PetscLayoutGetRange(layoutA2, &rsA2, nullptr));
        PetscCall(PetscLayoutDestroy(&layoutA2));
        PetscCall(PetscMalloc1(di_row[2], &locrA2));
        PetscCall(PetscMalloc1(di_col[2], &loccA2));
        for(PetscInt r = 0; r < di_row[2]; ++r){
            locrA2[r] = r+rsA2;
        }
        for(PetscInt c = 0; c < di_col[2]; ++c){
            loccA2[c] = c+rsA2;
        }
        PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, di_row[2], locrA2, PETSC_OWN_POINTER, &rowmA2));
        PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, di_col[2], loccA2, PETSC_OWN_POINTER, &colmA2));
        PetscCall(ISLocalToGlobalMappingDestroy(&rowmA2));
        PetscCall(ISLocalToGlobalMappingDestroy(&colmA2));


        // make pointers of all of these things
        // make a


        // submat
        PetscCall(PetscCalloc1(DD, &Ai));
        for(PetscInt i = 0; i < D; ++i){
            PetscCall(MatCreate(PETSC_COMM_WORLD, &Ai[i*(D+1)]));
            PetscCall(MatSetSizes(Ai[i*(D+1)], di_row[i], di_col[i], PETSC_DECIDE, PETSC_DECIDE));
        }
        for(PetscInt i = 0; i < D; ++i){
            PetscCall(MatSetUp(Ai[i*(D+1)]));
        }
        PetscCall(MatSetPreallocationCOO_SubMatrix(A1_col, A1_row, Ai[0]));
        PetscCall(MatSetPreallocationCOO_SubMatrix(A2_col, A2_row, Ai[4]));
        PetscCall(MatSetPreallocationCOO_SubMatrix(A3_col, A3_row, Ai[8]));
        // mat
        PetscCall(MatNestSetSubMats(A, D, nullptr, D, nullptr, Ai));
        for(PetscInt i = 0; i < DD; ++i){
            PetscCall(MatDestroy(&Ai[i]));
        }
        PetscCall(PetscFree(Ai));
    }
    PetscCall(MatSetLocal_SubMatrix(0, 0, A1_val, A));
    PetscCall(MatSetLocal_SubMatrix(1, 1, A2_val, A));
    PetscCall(MatSetLocal_SubMatrix(2, 2, A3_val, A));
    //
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatViewFromOptions(A, nullptr, "-A_view"));
    PetscCall(MatDestroy(&A));


    PetscCall(Finalize());
    PetscFunctionReturn(PETSC_SUCCESS);
}

// -mat_type nest -A_view -mat_view_nest_sub -malloc_dump -malloc_debug -malloc_dump
// https://github.com/FilipeFT24/PETSc-repo/blob/master/ex81a.cpp
// https://gitlab.com/petsc/petsc/-/blob/fccfa4b43f74198ab890b077de64eee091c20710/src/ksp/ksp/tutorials/ex85.c
// https://gitlab.inria.fr/aerosol/aerosol/-/blob/fft_phd/lib/include/common/SMatrixPETSc.hpp?ref_type=heads