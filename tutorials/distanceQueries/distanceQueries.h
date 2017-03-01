// ======================================================================== //
// Copyright 2009-2017 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

/*! \file distanceQueries.h C99/Fortran style API to performing
    distance queries of the sort "given point P and triangles T[],
    find closest point P' \in t, among all triangle t in T */

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif
  
  typedef void *distance_query_scene;

  /*! create a new 'general' triangle mesh that uses int32's ('i') for
      vertex indices, and single-precision floats ('f') for the vertex
      coordinates. Using the 'stride', and having potentially separate
      arrays for x, y, and z coordinates should allow for all kinds of
      data layouts, from C/C++ style arrays of structs, to
      fortran-style 'one array per component'.The 'stride' is measured
      in scalar types, so '1' means '4 bytes' when dealing with
      floats, '8 bytes' when dealing with doubles, etc.

      Example 1: for the case with different arrays per component:
      
      float vertex_x[N_VERTS];
      float vertex_y[N_VERTS];
      float vertex_z[N_VERTS];
      ...
      use
      t = rtdqNewTriangleMeshfi(vertex_x,vertex_y,vertex_z,1, ....
      
      (a stride of '1' says that the next vertex's coordiante is one
      scalar behind the previous one).

      
      Example 2: For a structure of arrays, simply have the x/y/z
      pointer's point to the respective component of the first
      element, and specify a stride consistent with the number of
      scalars per struct. E.g.,

      std::vector<vec3f> vertex;
      t = rtdqNewTriangleMeshfi(&vertex[0].x,&vertex[0].y,&vertex[0].z,3, ...

  */
  distance_query_scene rtdqNewTriangleMeshfi(const float   *vertex_x,
                                             const float   *vertex_y,
                                             const float   *vertex_z,
                                             const size_t   vertex_stride,
                                             const int32_t *index_x,
                                             const int32_t *index_y,
                                             const int32_t *index_z,
                                             const size_t   index_stride,
                                             const size_t    numTriangles);

  /*! create a new 'general' triangle mesh that uses int32's ('i') for
      vertex indices, and double-precision floats ('d') for the vertex
      coordinates. Using the 'stride', and having potentially separate
      arrays for x, y, and z coordinates should allow for all kinds of
      data layouts, from C/C++ style arrays of structs, to
      fortran-style 'one array per component'. The 'stride' is
      measured in scalar types, so '1' means '4 bytes' when dealing
      with floats, '8 bytes' when dealing with doubles, etc.

  */
  distance_query_scene rtdqNewTriangleMeshdi(const double  *vertex_x,
                                             const double  *vertex_y,
                                             const double  *vertex_z,
                                             const size_t   vertex_stride,
                                             const int32_t *index_x,
                                             const int32_t *index_y,
                                             const int32_t *index_z,
                                             const size_t   index_stride,
                                             const size_t   numTriangles);
  
  /*! destroy a scene created with rtdqNew...() */
  void rtdqDestroy(distance_query_scene scene);
  
  /*! for each point in in_query_point[] array, find the closest point
    P' among all the triangles in the given scene, and store position
    of closest triangle point, primID of the triangle that this point
    belongs to, and distance to that closest point, in the
    corresponding output arrays.
    
    for fullest flexibilty we are passing individual base pointers and
    strides for every individual array member, thus allowing to use
    both C++-style array of structs as well as fortran-suyle list of
    arrays data layouts
  */
  void rtdqComputeClosestPoints(distance_query_scene scene,
                                float   *out_closest_point_pos_x,
                                float   *out_closest_point_pos_y,
                                float   *out_closest_point_pos_z,
                                size_t   out_closest_point_pos_stride,
                                float   *out_closest_point_dist,
                                size_t   out_closest_point_dist_stride,
                                int32_t *out_closest_point_primID,
                                size_t   out_closest_point_primID_stride,
                                const float *in_query_point_x,
                                const float *in_query_point_y,
                                const float *in_query_point_z,
                                const size_t in_query_point_stride,
                                const size_t numQueryPoints);
  
#ifdef __cplusplus
}
#endif
