cmake_minimum_required(VERSION 3.20)
project(DerivativesTools VERSION 1.0.0 LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create the library target (static or shared)
add_library(DerivativesTools
        src/models.cpp
        )

# Include directories for public headers
target_include_directories(DerivativesTools PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )

# Create an executable for demonstration/testing
add_executable(DerivativesToolsExample main.cpp)
target_link_libraries(DerivativesToolsExample PRIVATE DerivativesTools)

# Installation rules
include(GNUInstallDirs)
install(TARGETS DerivativesTools
        EXPORT DerivativesToolsTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}    # For shared libraries
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}    # For static libraries
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}    # For executables (if applicable)
        )
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export configuration for find_package
install(EXPORT DerivativesToolsTargets
        FILE DerivativesToolsConfig.cmake
        NAMESPACE DerivativesTools::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/DerivativesTools
        )

# Optional: Enable testing
enable_testing()
