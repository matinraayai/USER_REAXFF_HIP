cmake_minimum_required(VERSION 3.10)
project(USER_REAXFF_HIP CXX)
# Add HIP
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm /usr/bin)
find_package(hip)
set(CMAKE_CXX_COMPILER "hipcc")
# Add MPI
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# Debug flags
set(CMAKE_CXX_FLAGS "-g -O0")

# Find all files
file(GLOB LAMMPS_FILES "fmt/*.h" "*.h" "*.cpp")

# Create executable
add_definitions(-DLAMMPS_REAX -DHAVE_HIP)
add_executable(lmp_hip ${LAMMPS_FILES})
# Link with HIP
target_link_libraries(lmp_hip hip::device ${MPI_CXX_LIBRARIES})