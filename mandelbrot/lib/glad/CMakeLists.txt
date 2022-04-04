
add_library(glad glad.c glad.h)
target_include_directories(glad PUBLIC .)
set_target_properties(glad PROPERTIES LINKER_LANGUAGE CXX)
