## ======================================================================== ##
## Copyright 2009-2017 Intel Corporation                                    ##
##                                                                          ##
## Licensed under the Apache License, Version 2.0 (the "License");          ##
## you may not use this file except in compliance with the License.         ##
## You may obtain a copy of the License at                                  ##
##                                                                          ##
##     http://www.apache.org/licenses/LICENSE-2.0                           ##
##                                                                          ##
## Unless required by applicable law or agreed to in writing, software      ##
## distributed under the License is distributed on an "AS IS" BASIS,        ##
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. ##
## See the License for the specific language governing permissions and      ##
## limitations under the License.                                           ##
## ======================================================================== ##

INCLUDE(tutorial)

ADD_LIBRARY(embree_distance_queries
  distanceQueries.cpp
  )
TARGET_LINK_LIBRARIES(embree_distance_queries
  embree
  )

ADD_EXECUTABLE(distanceQueryExample
  example.cpp
  )
TARGET_LINK_LIBRARIES(distanceQueryExample
  embree_distance_queries
  )

SET_PROPERTY(TARGET distanceQueryExample PROPERTY FOLDER examples/stanceQueries)
INSTALL(TARGETS distanceQueryExample DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT examples)