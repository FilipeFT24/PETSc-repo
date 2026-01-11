#include <petscksp.h>

PetscErrorCode MatSetLocalSubMatrix(PetscInt m, PetscInt n, const PetscInt *a_row, const PetscInt *a_col, const PetscScalar *a_val, Mat A) {
    IS row, col;
    Mat a;
    PetscFunctionBeginUser;
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, m, a_row, PETSC_COPY_VALUES, &row));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, n, a_col, PETSC_COPY_VALUES, &col));
    PetscCall(MatGetLocalSubMatrix(A, row, col, &a));
    for(PetscInt i = 0; i < m; ++i){
        for(PetscInt j = 0; j < n; ++j){
            PetscCall(MatSetValuesLocal(a, 1, &i, 1, &j, &a_val[i*n+j], INSERT_VALUES));
        }
    }
    PetscCall(MatRestoreLocalSubMatrix(A, row, col, &a));
    PetscCall(ISDestroy(&row));
    PetscCall(ISDestroy(&col));
    PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char **argv){
#ifdef WITH_SLEPC
    PetscCall(SlepcInitialize(&argc, &argv, nullptr, nullptr));
#else
    PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
#endif
    PetscCall(PetscLogDefaultBegin());

    PetscInt  m = 5, n = 5, M = PETSC_DETERMINE, N = PETSC_DETERMINE, rStart, cStart;
    PetscBool is_nest;
    Mat       G;

    PetscCall(MatCreate(PETSC_COMM_WORLD, &G));
    PetscCall(PetscSplitOwnership(PETSC_COMM_WORLD, &m, &M));
    PetscCall(PetscSplitOwnership(PETSC_COMM_WORLD, &n, &N));
    PetscCall(MatSetSizes(G, m, n, M, N));
    PetscCall(MatSetFromOptions(G));
    PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(G), MATNEST, &is_nest));
    PetscCall(MatGetOwnershipRange(G, &rStart, nullptr));
    PetscCall(MatGetOwnershipRangeColumn(G, &cStart, nullptr));





    ISLocalToGlobalMapping rowMap, colMap;
    PetscInt *locRows, *locCols;






    if (is_nest) {
        Mat submat[4];
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
        PetscCall(
            ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, 3, locRowsA, PETSC_OWN_POINTER, &
                rowMapA));
        PetscCall(
            ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, 3, locColsA, PETSC_OWN_POINTER, &
                colMapA));
        PetscCall(PetscLayoutCreate(PETSC_COMM_WORLD, &layoutD));
        PetscCall(PetscLayoutSetLocalSize(layoutD, 2));
        PetscCall(PetscLayoutSetUp(layoutD));
        PetscCall(PetscLayoutGetRange(layoutD, &rStartD, nullptr));
        PetscCall(PetscLayoutDestroy(&layoutD));
        PetscCall(PetscMalloc1(2, &locRowsD));
        for (PetscInt r = 0; r < 2; ++r) locRowsD[r] = r + rStartD;
        PetscCall(PetscMalloc1(2, &locColsD));
        for (PetscInt c = 0; c < 2; ++c) locColsD[c] = c + rStartD;
        PetscCall(
            ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, 2, locRowsD, PETSC_OWN_POINTER, &
                rowMapD));
        PetscCall(
            ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, 2, locColsD, PETSC_OWN_POINTER, &
                colMapD));

        PetscCall(MatCreate(PETSC_COMM_WORLD, &submat[0]));
        PetscCall(MatSetSizes(submat[0], 3, 3, PETSC_DETERMINE, PETSC_DETERMINE));
        PetscCall(MatSetType(submat[0], MATAIJ));
        PetscCall(MatSetLocalToGlobalMapping(submat[0], rowMapA, colMapA));
        PetscCall(MatCreate(PETSC_COMM_WORLD, &submat[1]));
        PetscCall(MatSetSizes(submat[1], 3, 2, PETSC_DETERMINE, PETSC_DETERMINE));
        PetscCall(MatSetType(submat[1], MATAIJ));
        PetscCall(MatSetLocalToGlobalMapping(submat[1], rowMapA, colMapD));
        PetscCall(MatCreate(PETSC_COMM_WORLD, &submat[2]));
        PetscCall(MatSetSizes(submat[2], 2, 3, PETSC_DETERMINE, PETSC_DETERMINE));
        PetscCall(MatSetType(submat[2], MATAIJ));
        PetscCall(MatSetLocalToGlobalMapping(submat[2], rowMapD, colMapA));
        PetscCall(MatCreate(PETSC_COMM_WORLD, &submat[3]));
        PetscCall(MatSetSizes(submat[3], 2, 2, PETSC_DETERMINE, PETSC_DETERMINE));
        PetscCall(MatSetType(submat[3], MATAIJ));
        PetscCall(MatSetLocalToGlobalMapping(submat[3], rowMapD, colMapD));
        for (PetscInt i = 0; i < 4; ++i)
            PetscCall(MatSetUp(submat[i]));
        PetscCall(MatNestSetSubMats(G, 2, nullptr, 2, nullptr, submat));
        for (PetscInt i = 0; i < 4; ++i)
            PetscCall(MatDestroy(&submat[i]));

        PetscCall(ISLocalToGlobalMappingDestroy(&rowMapA));
        PetscCall(ISLocalToGlobalMappingDestroy(&colMapA));
        PetscCall(ISLocalToGlobalMappingDestroy(&rowMapD));
        PetscCall(ISLocalToGlobalMappingDestroy(&colMapD));
    }
    PetscCall(PetscMalloc1(m, &locRows));
    for (PetscInt r = 0; r < m; ++r) locRows[r] = r + rStart;
    PetscCall(PetscMalloc1(n, &locCols));
    for (PetscInt c = 0; c < n; ++c) locCols[c] = c + cStart;
    PetscCall(
        ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, m, locRows, PETSC_OWN_POINTER, &rowMap));
    PetscCall(
        ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, n, locCols, PETSC_OWN_POINTER, &colMap));
    PetscCall(MatSetLocalToGlobalMapping(G, rowMap, colMap));
    PetscCall(ISLocalToGlobalMappingDestroy(&rowMap));
    PetscCall(ISLocalToGlobalMappingDestroy(&colMap));




    PetscInt A_row[] = {0, 1, 2};
    PetscInt A_col[] = {0, 1, 2};
    PetscInt B_row[] = {0, 1, 2};
    PetscInt B_col[] = {3, 4};
    PetscInt C_row[] = {3, 4};
    PetscInt C_col[] = {0, 1, 2};
    PetscInt D_row[] = {3, 4};
    PetscInt D_col[] = {3, 4};
    // Values
    PetscScalar A_val[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    PetscScalar B_val[] = {0, 1, 2, 3, 4, 5};
    PetscScalar C_val[] = {0, 1, 2, 3, 4, 5};
    PetscScalar D_val[] = {0, 1, 2, 3};
    //
    m = PETSC_STATIC_ARRAY_LENGTH(A_row);
    n = PETSC_STATIC_ARRAY_LENGTH(A_col);
    PetscCall(MatSetLocalSubMatrix(m, n, A_row, A_col, A_val, G));
    m = PETSC_STATIC_ARRAY_LENGTH(B_row);
    n = PETSC_STATIC_ARRAY_LENGTH(B_col);
    PetscCall(MatSetLocalSubMatrix(m, n, B_row, B_col, B_val, G));
    m = PETSC_STATIC_ARRAY_LENGTH(C_row);
    n = PETSC_STATIC_ARRAY_LENGTH(C_col);
    PetscCall(MatSetLocalSubMatrix(m, n, C_row, C_col, C_val, G));
    m = PETSC_STATIC_ARRAY_LENGTH(D_row);
    n = PETSC_STATIC_ARRAY_LENGTH(D_col);
    PetscCall(MatSetLocalSubMatrix(m, n, D_row, D_col, D_val, G));
    //
    PetscCall(MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY));
    PetscCall(MatViewFromOptions(G, nullptr, "-G_view"));
    PetscCall(MatDestroy(&G));
    PetscCall(PetscFinalize());
    return 0;
}
