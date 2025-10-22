cmake_minimum_required(VERSION 3.24)

# Set the project name
file(STRINGS project.txt PRJ_NAME LIMIT_COUNT 1)
project(${PRJ_NAME})

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# CXX flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall") # -Wextra -Wpedantic
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    # add_definitions(-D_GLIBCXX_DEBUG)             # Note: This needs all other dependencies to also be built in debug mode
    # add_definitions(-DLIBCXX_ENABLE_DEBUG_MODE)
    add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_NAN)
else()
    add_definitions(-DNDEBUG)
    add_definitions(-DEIGEN_NO_DEBUG)
endif()

# Force coloured compiler output
SET(CMAKE_COLOR_DIAGNOSTICS ON)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")          # match Clang or AppleClang
    # Support colours for ninja + clang under MSYS2
    # (see https://github.com/msys2/MINGW-packages/issues/2988#issuecomment-1425591134)
    add_compile_options(-fansi-escape-codes)
    add_link_options(-fansi-escape-codes)
endif()

# Workaround for Qt5 being unlinked under homebrew
if(NOT DEFINED ENV{Qt5_DIR} OR ENV{Qt5_DIR} STREQUAL "Qt5_DIR-NOTFOUND")
    execute_process(COMMAND brew --prefix qt@5 OUTPUT_VARIABLE PREFIX_QT5 OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(ENV{Qt5_DIR} ${PREFIX_QT5}/lib/cmake/Qt5)
endif()

include(FetchContent)

FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG v4.3.11
    GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(nanobench)

find_package(doctest REQUIRED)
find_package(Boost 1.74 REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(autodiff 1.0 REQUIRED)
find_package(OpenCV 4.7 REQUIRED)
find_package(VTK 8.2 REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(Doxygen QUIET)     # optional

if(onnxruntime_FOUND)
    add_definitions(-DWITH_ONNX)
else()
    message(STATUS "ONNX Runtime not found. Building without ONNX Runtime support.")
endif()

option(BUILD_DOCUMENTATION "Build documentation" OFF)

if(BUILD_DOCUMENTATION)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        # Set input and output files
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

        # Request to configure the file
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

        # Get all source files
        file(GLOB_RECURSE DOC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*)

        # Add doxygen target
        doxygen_add_docs(docs
            ${DOC_SOURCES} # Can't be a directory or wildcards if USE_STAMP_FILE is used (https://cmake.org/cmake/help/latest/module/FindDoxygen.html)
            CONFIG_FILE ${DOXYGEN_OUT}
            COMMENT "Generating API documentation with Doxygen"
            USE_STAMP_FILE
        )
    else()
        message(STATUS "Doxygen not found, documentation will not be built")
    endif()
endif()

if(VTK_VERSION VERSION_LESS "8.90.0")
    include(${VTK_USE_FILE})
else()
    include_directories(${VTK_PREFIX_PATH}/include/vtk-${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION})
    include_directories(${VTK_PREFIX_PATH}/include/vtk) # Support VTK 9.x under MSYS2
endif()

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

# Target names
set(TARGET_LIB common)
set(TARGET_BIN ${PROJECT_NAME})
set(TARGET_TEST tests)

# Libraries
set(LIBS
    ${Boost_LIBRARIES}
    Eigen3::Eigen
    autodiff::autodiff
    ${OpenCV_LIBS}
    ${VTK_LIBRARIES}
    nanobench
)

if(onnxruntime_FOUND)
    list(APPEND LIBS onnxruntime::onnxruntime)
endif()

# Source files
set(DIR_SRC src)
set(DIR_TEST_SRC test/src)

# Application
file(GLOB SRC_BIN CONFIGURE_DEPENDS ${DIR_SRC}/*.cpp ${DIR_SRC}/*.h ${DIR_SRC}/*.hpp)

# Static library
set(SRC_LIB ${SRC_BIN})
list(FILTER SRC_LIB EXCLUDE REGEX ".*main.cpp$")

# Unit tests
file(GLOB SRC_TEST CONFIGURE_DEPENDS ${DIR_TEST_SRC}/*.cpp ${DIR_TEST_SRC}/*.h ${DIR_TEST_SRC}/*.hpp)

# Targets

# Static library
if(SRC_LIB)
    add_library(${TARGET_LIB} ${SRC_LIB})
    target_link_libraries(${TARGET_LIB} ${LIBS})
endif()

# Application
add_executable(${TARGET_BIN} src/main.cpp)
if(SRC_LIB)
    target_link_libraries(${TARGET_BIN} ${TARGET_LIB})
endif()
target_link_libraries(${TARGET_BIN} ${LIBS})

if(VTK_VERSION VERSION_GREATER_EQUAL "8.90.0")
    vtk_module_autoinit(TARGETS ${TARGET_BIN} MODULES ${VTK_LIBRARIES})
endif()

# Add documentation dependency if applicable
if(BUILD_DOCUMENTATION AND DOXYGEN_FOUND)
    add_dependencies(${TARGET_BIN} docs)
endif()

# Unit tests
add_executable(${TARGET_TEST} ${SRC_TEST})
if(SRC_LIB)
    target_link_libraries(${TARGET_TEST} ${TARGET_LIB})
endif()
target_link_libraries(${TARGET_TEST} ${LIBS})
target_link_libraries(${TARGET_TEST} doctest::doctest)

# Run unit tests after building executables
add_custom_target(run_tests ALL
    COMMAND ${TARGET_TEST} --no-intro --force-colors=true
    DEPENDS ${TARGET_TEST}
    DEPENDS ${TARGET_BIN}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running tests"
)

set_property(GLOBAL PROPERTY TARGET_MESSAGES OFF)

