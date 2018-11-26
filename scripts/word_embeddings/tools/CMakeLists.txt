cmake_minimum_required(VERSION 3.9..3.13)

# * Set and configure build types
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

project(EmbeddingsToolkit VERSION 1.0 LANGUAGES CXX)

# * Update submodules as needed
find_package(Git QUIET)
if(GIT_FOUND)
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

# * Compiler flags
# -march=native or similar is required by __sync_bool_compare_and_swap_16 in growt
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# * Executables
add_subdirectory(extern/cnpy EXCLUDE_FROM_ALL)

# ** vocab_count
add_executable(vocab_count vocab_count.cc)
target_include_directories(vocab_count PUBLIC extern/sparsepp)
target_include_directories(vocab_count PUBLIC extern/CLI11/include)
target_link_libraries(vocab_count PUBLIC stdc++fs)
target_link_libraries(vocab_count PUBLIC pthread)

# ** cooccur
add_executable(cooccur cooccur.cc)
target_link_libraries(cooccur PUBLIC pthread)
target_include_directories(cooccur PUBLIC extern/sparsepp)
target_include_directories(cooccur PUBLIC extern/xxHash)
target_include_directories(cooccur PUBLIC extern/growt)
target_include_directories(cooccur PUBLIC extern/range-v3/include)
target_include_directories(cooccur PUBLIC extern/cnpy)
target_include_directories(cooccur PUBLIC extern/CLI11/include)
target_link_libraries(cooccur PUBLIC stdc++fs)
target_link_libraries(cooccur PUBLIC cnpy)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cooccur PUBLIC OpenMP::OpenMP_CXX)
endif()

