cmake_minimum_required(VERSION 3.10)

project(main CXX)

set(PETSC_DIR "/home/fft/petsc")
set(PETSC_ARCH "arch-linux-c-debug")

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC
    "${PETSC_DIR}/include"
    "${PETSC_DIR}/${PETSC_ARCH}/include"
    "/usr/lib/x86_64-linux-gnu/openmpi/include"
    "/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    "${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so"
    "/usr/lib/x86_64-linux-gnu/libmpi.so"
    m
    dl
    pthread
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
)

#https://gitlab.com/petsc/petsc/-/tree/214548ce03bf66e5974d3592a95603a9db2deb99/src