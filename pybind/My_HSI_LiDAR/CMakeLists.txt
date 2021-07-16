cmake_minimum_required(VERSION 3.0)
project(My_HSI_LiDAR)

find_package(pybind11)

#set(PYBIND11_PYTHON_VERSION 3.5)

# Find installed Open3D, this imports
#   Open3D_INCLUDE_DIRS
#   Open3D_LIBRARIES
#   Open3D_LIBRARY_DIRS
#   Open3D_C_FLAGS
#   Open3D_CXX_FLAGS
#   Open3D_EXE_LINKER_FLAGS
SET(CMAKE_INSTALL_PREFIX /home/jxd/install)

if(WIN32)
    find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/CMake)
else()
    find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/cmake)
    list(APPEND Open3D_LIBRARIES dl)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Open3D_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")

# Set OS-specific things here
if(WIN32)
elseif(CYGWIN)
elseif(APPLE)
elseif(UNIX)
        add_definitions(-DUNIX)
        add_compile_options(-Wno-deprecated-declarations)
        add_compile_options(-Wno-unused-result)
    add_definitions(-O3)
endif(WIN32)

find_package( OpenCV REQUIRED )

include_directories("./" "/usr/local/include/Open3D/3rdparty")
link_directories(${Open3D_LIBRARY_DIRS})


add_library(${PROJECT_NAME} MODULE main.cpp ndarray_converter.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::module ${Open3D_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")
target_include_directories(${PROJECT_NAME} PUBLIC ${Open3D_INCLUDE_DIRS} )
