cmake_minimum_required (VERSION 2.8.11)
project (NgBem)


find_package ( NGSolve CONFIG REQUIRED )

Set(CMAKE_CXX_COMPILER "ngscxx" CACHE STRING "" FORCE)
#set (CMAKE_CXX_STANDARD 14)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/ ${CMAKE_SOURCE_DIR}/cmake/GreatCMakeCookOff/)


find_package(Bempp 3)

option(USE_PYTHON On)

find_package( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )

#find_package(TBB REQUIRED)
#find_package(MPI)


set(Python_ADDITIONAL_VERSIONS 3.5)
find_package(PythonLibs 3 REQUIRED)
message("Libs of Python: " ${PYTHON_LIBRARIES} )


if (USE_PYTHON)
  find_path(PYBIND_INCLUDE_DIR pybind11/pybind11.h ${NETGEN_INTERNAL_INCLUDE_DIR}/../../external_dependencies/pybind11/include)
  if( NOT PYBIND_INCLUDE_DIR )
    message(FATAL_ERROR "Could NOT find pybind11!")
  endif( NOT PYBIND_INCLUDE_DIR )
  message("-- Found Pybind11: ${PYBIND_INCLUDE_DIR}")
  include_directories(BEFORE ${PYBIND_INCLUDE_DIR})
  install(DIRECTORY ${PYBIND_INCLUDE_DIR}/pybind11 DESTINATION include COMPONENT ngsolve_devel)
  install(FILES ${PYBIND_INCLUDE_DIR}/../LICENSE DESTINATION include/pybind11 COMPONENT ngsolve_devel)
  set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${PROJECT_SOURCE_DIR}/cmake_modules/python")
  set(PYTHON_VERSION "3" CACHE STRING "Python version (only Python >= 3.0 supported)")
  set(Python_ADDITIONAL_VERSIONS 3.5)
  if( PYTHON_VERSION VERSION_LESS 3 )
    message(FATAL_ERROR "NGSolve supports only Python 3")
  endif( PYTHON_VERSION VERSION_LESS 3 )
  find_package(PythonInterp ${PYTHON_VERSION} REQUIRED)
  find_package(PythonLibs ${PYTHON_VERSION}  REQUIRED)
  add_definitions(-DNGS_PYTHON)

  include_directories(${PYTHON_INCLUDE_DIRS})
  set(PYTHON_LIBS "${PYTHON_LIBRARIES}")
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib(1,0,''))" OUTPUT_VARIABLE PYTHON_PACKAGES_INSTALL_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif (USE_PYTHON)


include(LookUp-GreatCMakeCookOff)


include(PythonPackageLookup)
include(PythonInstall)
include(PythonModule)

link_directories(${NETGENDIR}/../lib)

lookup_python_package(Cython VERSION 0.21 REQUIRED PATH "${EXTERNAL_ROOT}/python")

OPTION(PARALLEL_ASSEMBLY "Try to assebmble the Frequency domain operators simultaneously" On)
OPTION(WITH_MPI "Try to use MPI multiprocessing" ${MPI_C_FOUND})


SET(BEMPP_CYTHON_DIR ${Bempp_DIR}/python_package/bempp)

include_directories("${BEMPP_INCLUDE_DIRS}")
include_directories(${Bempp_INCLUDE_DIR})


find_path(BEMPP_CYTHON_INCLUDE_DIR bempp/core/__init__.pxd HINTS ${Bempp_DIR}/python_package/bempp/include  ${Bempp_Cython_DIR})

include_directories(${BEMPP_CYTHON_INCLUDE_DIR})


set(CMAKE_CXX_FLAGS "-Wno-narrowing  ") #-DEIGEN_INITIALIZE_MATRICES_BY_NAN



if( WITH_MPI)
  include_directories(${MPI_C_INCLUDE_PATH})
endif (WITH_MPI)

#include_directories(${BEMPP_SRC_DIR}/cython/)
include_directories(${CMAKE_SOURCE_DIR})
SET( LIBRARY_SRCS blockmatrix.cpp ng_trace_space.cpp ng_grid_factory.cpp blockmatrix.cpp )



add_library(ngbem SHARED ${LIBRARY_SRCS} )
target_link_libraries(ngbem  ${TBB_LIBRARIES}  ngsolve ngstd ngfem ngbla ngla ngcomp interface ${PYTHON_LIBRARIES} libbempp )

add_python_module(pyngbem_common pyngbem.cpp  CPP  LIBRARIES ngsolve ngstd ngfem ngbla ngcomp interface  ${PYTHON_LIBRARIES} libbempp ngbem )
add_python_module(pyngbem __init__.py space.pyx blockmatrix.py  CPP  LIBRARIES ngsolve ngstd ngfem ngbla ngcomp interface  ${PYTHON_LIBRARIES} libbempp ngbem )


#add_python_module(ngbem _SRCS} CPP  LIBRARIES ngsolve ngstd ngfem ngbla interface  ${PYTHON_LIBRARIES} libbempp)




include_directories(${TBB_INCLUDE_DIRS})

include_directories(${CMAKE_BINARY_DIR})

