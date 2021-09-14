cmake_minimum_required(VERSION 3.10)
project(USER_REAXFF_HIP CXX)
# Add HIP
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm /usr/bin)
find_package(hip)
set(CMAKE_C_COMPILER "hipcc")
set(CMAKE_CXX_COMPILER "hipcc")
#set(CMAKE_CUDA_COMPILER "hipcc")
#set(MPI_CXX_COMPILER "mpic++")
#set(CMAKE_CXX_LINK_EXECUTABLE "hipcc")
#set(CMAKE_EXE_LINKER_FLAGS "--driver-mode=g++ -L"/opt/rocm-4.3.0/hip/lib" -O3 -lgcc_s -lgcc -lpthread -lm -lrt -lmpi_cxx -lmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lopen-pal")
set(CMAKE_EXE_LINKER_FLAGS "-g --driver-mode=g++ -L'/opt/rocm-4.3.0/hip/lib' -L/usr/lib/x86_64-linux-gnu -lgcc_s -lgcc -lpthread -lm -lrt -lmpichcxx -lmpich -Wl,-Bsymbolic-functions -Wl,-z,relro")
#set(CMAKE_CXX_FLAGS "-I"/opt/rocm-4.3.0/hip/include" -I"/opt/rocm-4.3.0/llvm/bin/../lib/clang/13.0.0" -I/opt/rocm-4.3.0/hsa/include")
set(CMAKE_CXX_FLAGS "-g -O0")
# Add MPI
#find_package(MPI REQUIRED)
include_directories(SYSTEM "/usr/include/mpich" "/opt/rocm-4.3.0/hip/include" "/opt/rocm-4.3.0/llvm/bin/../lib/clang/13.0.0" "/opt/rocm-4.3.0/hsa/include")

# Find all files
file(GLOB LAMMPS_FILES
        "fmt/*.h"
        "*.h"
        "*.cpp"
        )


# Create executable
add_definitions(-DLAMMPS_REAX -DHAVE_HIP)
add_executable(lmp_hip ${LAMMPS_FILES})
# Link with HIP
target_link_libraries(lmp_hip hip::device)