# CMakeList.txt : CMake project for MasterThesis, include source and define
# project specific logic here.
#

# Add source to this project's executable.

file(GLOB ANNIncludeSources RELATIVE "ANN/include/ANN" "*.h")
file(GLOB ANNSources RELATIVE "ANN/src" "*.cpp")

add_subdirectory("pybind11")

add_executable (NNPNET "main.cpp" "main.h" "Graph.h"  "Utils.h" "LayoutMethods/PivotMDS.h"  "LayoutMethods/tsNET.h"  "LayoutMethods/NNPNET.h"  "LayoutMethods/NNPNET.cpp"   "ANN/include/ANN/ANN.h" "ANN/include/ANN/ANNperf.h" "ANN/include/ANN/ANNx.h" "ANN/src/ANN.cpp" "ANN/src/bd_fix_rad_search.cpp" "ANN/src/bd_pr_search.cpp" "ANN/src/bd_search.cpp" "ANN/src/bd_tree.cpp" "ANN/src/brute.cpp" "ANN/src/kd_dump.cpp" "ANN/src/kd_fix_rad_search.cpp" "ANN/src/kd_pr_search.cpp" "ANN/src/kd_search.cpp" "ANN/src/kd_split.cpp" "ANN/src/kd_tree.cpp" "ANN/src/kd_util.cpp" "ANN/src/perf.cpp" "Threading.h" "Threading.cpp" "LayoutMethods/tsNETTree/sptree.h" "LayoutMethods/tsNETTree/sptree.cpp" "Smoothing.h" "Procrustes.h" "Procrustes.cpp" "Stress.h" "NeighborhoodPreservation.h")

target_include_directories(NNPNET PRIVATE "ANN/include")

target_include_directories(NNPNET PRIVATE "pybind11/include")
set(PYBIND11_FINDPYTHON ON)
target_link_libraries(NNPNET PRIVATE pybind11::embed)

# python library

pybind11_add_module(NNP_NET "main.cpp" "main.h" "Graph.h"  "Utils.h" "LayoutMethods/PivotMDS.h"  "LayoutMethods/tsNET.h"  "LayoutMethods/NNPNET.h"  "LayoutMethods/NNPNET.cpp"   "ANN/include/ANN/ANN.h" "ANN/include/ANN/ANNperf.h" "ANN/include/ANN/ANNx.h" "ANN/src/ANN.cpp" "ANN/src/bd_fix_rad_search.cpp" "ANN/src/bd_pr_search.cpp" "ANN/src/bd_search.cpp" "ANN/src/bd_tree.cpp" "ANN/src/brute.cpp" "ANN/src/kd_dump.cpp" "ANN/src/kd_fix_rad_search.cpp" "ANN/src/kd_pr_search.cpp" "ANN/src/kd_search.cpp" "ANN/src/kd_split.cpp" "ANN/src/kd_tree.cpp" "ANN/src/kd_util.cpp" "ANN/src/perf.cpp" "Threading.h" "Threading.cpp" "LayoutMethods/tsNETTree/sptree.h" "LayoutMethods/tsNETTree/sptree.cpp" "Smoothing.h" "Procrustes.h" "Procrustes.cpp")
target_include_directories(NNP_NET PRIVATE "ANN/include")

target_include_directories(NNP_NET PRIVATE "pybind11/include")
set(PYBIND11_FINDPYTHON ON)
target_link_libraries(NNP_NET PRIVATE pybind11::embed)
install(TARGETS NNP_NET DESTINATION .)


if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" AND CMAKE_BUILD_TYPE MATCHES "Release")
   target_compile_options(NNPNET PRIVATE /Zi)

   # Tell linker to include symbol data
    set_target_properties(NNPNET PROPERTIES 
        LINK_FLAGS "/INCREMENTAL:NO /DEBUG /OPT:REF /OPT:ICF"
    )

    # Set file name & location
    set_target_properties(NNPNET PROPERTIES 
        COMPILE_PDB_NAME NNPNET 
        COMPILE_PDB_OUTPUT_DIR ${CMAKE_BINARY_DIR}
    )
endif()

if (MSVC)
  add_definitions(/bigobj)
endif()

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()

endif()

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET NNPNET PROPERTY CXX_STANDARD 17)
  set_property(TARGET NNP_NET PROPERTY CXX_STANDARD 17)
endif()

