#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "DerivativesTools::DerivativesTools" for configuration "Debug"
set_property(TARGET DerivativesTools::DerivativesTools APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(DerivativesTools::DerivativesTools PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libDerivativesTools.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS DerivativesTools::DerivativesTools )
list(APPEND _IMPORT_CHECK_FILES_FOR_DerivativesTools::DerivativesTools "${_IMPORT_PREFIX}/lib/libDerivativesTools.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
