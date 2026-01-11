#include <petscksp.h>

PetscErrorCode MatSetLocalSubMatrix(const PetscInt ib, const PetscInt jb, const PetscScalar* a_val, Mat A) {
    Mat a;
    PetscFunctionBeginUser;
    PetscCall(MatNestGetSubMat(A, ib, jb, &a));
    PetscCall(MatSetValuesCOO(a, a_val, INSERT_VALUES));
    PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char** argv){
#ifdef WITH_SLEPC
    PetscCall(SlepcInitialize(&argc, &argv, nullptr, nullptr));
#else
    PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
#endif
    PetscCall(PetscLogDefaultBegin());

    PetscInt    m = 5, n = 5, M = PETSC_DECIDE, N = PETSC_DECIDE, rStart, cStart;
    PetscBool   is_nest;
    PetscInt    A_row[] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    PetscInt    A_col[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    PetscInt    B_row[] = {0, 0, 1, 1, 2, 2};
    PetscInt    B_col[] = {0, 1, 0, 1, 0, 1};
    PetscInt    C_row[] = {0, 0, 0, 1, 1, 1};
    PetscInt    C_col[] = {0, 1, 0, 1, 0, 1};
    PetscInt    D_row[] = {0, 0, 1, 1};
    PetscInt    D_col[] = {0, 1, 0, 1};
    PetscScalar A_val[] = {0, 1, 2, 5, 6, 7, 10, 11, 12};
    PetscScalar B_val[] = {3, 4, 8, 9, 13, 14};
    PetscScalar C_val[] = {15, 16, 17, 20, 21, 22};
    PetscScalar D_val[] = {18, 19, 23, 24};
    Mat         G;

    PetscCall(MatCreate(PETSC_COMM_WORLD, &G));
    PetscCall(PetscSplitOwnership(PETSC_COMM_WORLD, &m, &M));
    PetscCall(PetscSplitOwnership(PETSC_COMM_WORLD, &n, &N));
    PetscCall(MatSetSizes(G, m, n, M, N));
    PetscCall(MatSetFromOptions(G));
    /*
    PetscCall(MatSetUp(G));
    */
    PetscCall(MatSetOption(G, MAT_ROW_ORIENTED, PETSC_TRUE)); // -1
    PetscCall(MatSetOption(G, MAT_STRUCTURALLY_SYMMETRIC, PETSC_TRUE)); // 2
    PetscCall(MatSetOption(G, MAT_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE)); // 4
    PetscCall(MatSetOption(G, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE)); // 6
    PetscCall(MatSetOption(G, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE)); // 11
    PetscCall(MatSetOption(G, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE)); // 16
    PetscCall(MatSetOption(G, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE)); // 17
    PetscCall(MatSetOption(G, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE)); // 19
    PetscCall(MatSetOption(G, MAT_SORTED_FULL, PETSC_TRUE)); // 23
    PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(G), MATNEST, &is_nest));
    PetscCall(MatGetOwnershipRange(G, &rStart, nullptr));
    PetscCall(MatGetOwnershipRangeColumn(G, &cStart, nullptr));


    if (is_nest) {
        Mat mat[4];
        PetscLayout layoutA, layoutD;
        PetscInt *locRowsA, *locColsA, *locRowsD, *locColsD, rStartA, rStartD;
        ISLocalToGlobalMapping rowMapA, colMapA, rowMapD, colMapD;

        PetscCall(PetscLayoutCreate(PETSC_COMM_WORLD, &layoutA));
        PetscCall(PetscLayoutSetLocalSize(layoutA, 3));
        PetscCall(PetscLayoutSetUp(layoutA));
        PetscCall(PetscLayoutGetRange(layoutA, &rStartA, nullptr));
        PetscCall(PetscLayoutDestroy(&layoutA));
        PetscCall(PetscMalloc1(3, &locRowsA));
        for (PetscInt r = 0; r < 3; ++r) locRowsA[r] = r + rStartA;
        PetscCall(PetscMalloc1(3, &locColsA));
        for (PetscInt c = 0; c < 3; ++c) locColsA[c] = c + rStartA;
        PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, 3, locRowsA, PETSC_OWN_POINTER, &rowMapA));
        PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, 3, locColsA, PETSC_OWN_POINTER, &colMapA));
        PetscCall(PetscLayoutCreate(PETSC_COMM_WORLD, &layoutD));
        PetscCall(PetscLayoutSetLocalSize(layoutD, 2));
        PetscCall(PetscLayoutSetUp(layoutD));
        PetscCall(PetscLayoutGetRange(layoutD, &rStartD, nullptr));
        PetscCall(PetscLayoutDestroy(&layoutD));
        PetscCall(PetscMalloc1(2, &locRowsD));
        for (PetscInt r = 0; r < 2; ++r) locRowsD[r] = r + rStartD;
        PetscCall(PetscMalloc1(2, &locColsD));
        for (PetscInt c = 0; c < 2; ++c) locColsD[c] = c + rStartD;
        PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, 2, locRowsD, PETSC_OWN_POINTER, &rowMapD));
        PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, 2, locColsD, PETSC_OWN_POINTER, &colMapD));


        for(auto & i : mat){
            PetscCall(MatCreate(PETSC_COMM_WORLD, &i));
        }
        PetscCall(MatSetSizes(mat[0], 3, 3, PETSC_DECIDE, PETSC_DECIDE));
        PetscCall(MatSetSizes(mat[1], 3, 2, PETSC_DECIDE, PETSC_DECIDE));
        PetscCall(MatSetSizes(mat[2], 2, 3, PETSC_DECIDE, PETSC_DECIDE));
        PetscCall(MatSetSizes(mat[3], 2, 2, PETSC_DECIDE, PETSC_DECIDE));
        for(auto & i : mat){
            PetscCall(MatSetUp(i));
        }
        PetscCall(MatNestSetSubMats(G, 2, nullptr, 2, nullptr, mat));
        //
        PetscCall(MatSetPreallocationCOO(mat[0], 9, A_row, A_col));
        PetscCall(MatSetPreallocationCOO(mat[1], 6, B_row, B_col));
        PetscCall(MatSetPreallocationCOO(mat[2], 6, C_row, C_col));
        PetscCall(MatSetPreallocationCOO(mat[3], 4, D_row, D_col));
        for(auto & i : mat){
            PetscCall(MatDestroy(&i));
        }

        PetscCall(ISLocalToGlobalMappingDestroy(&rowMapA));
        PetscCall(ISLocalToGlobalMappingDestroy(&colMapA));
        PetscCall(ISLocalToGlobalMappingDestroy(&rowMapD));
        PetscCall(ISLocalToGlobalMappingDestroy(&colMapD));
    }


    PetscCall(MatSetLocalSubMatrix(0, 0, A_val, G));
    PetscCall(MatSetLocalSubMatrix(0, 1, B_val, G));
    PetscCall(MatSetLocalSubMatrix(1, 0, C_val, G));
    PetscCall(MatSetLocalSubMatrix(1, 1, D_val, G));
    //
    PetscCall(MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY));
    PetscCall(MatViewFromOptions(G, nullptr, "-G_view"));
    PetscCall(MatDestroy(&G));
    PetscCall(PetscFinalize());
    PetscFunctionReturn(PETSC_SUCCESS);
}

// -mat_type nest -G_view -mat_view_nest_sub -malloc_dump -malloc_debug -malloc_dump