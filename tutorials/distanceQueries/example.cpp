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

#include "distanceQueries.h"
#include "../../common/math/vec3.h"

#include <vector>
#include <chrono>

#define USE_TBB 1

#if USE_TBB
#  include <tbb/parallel_for.h>

  template<typename TASK_T>
  inline void parallel_for(int nTasks, TASK_T&& fcn)
  {
    tbb::parallel_for(0, nTasks, 1, std::forward<TASK_T>(fcn));
  }
#endif

namespace embree {

  typedef Vec3<float> vec3f;
  typedef Vec3<int>   vec3i;
  
  extern "C" int main(int ac, char **av)
  {
    int numTriangles = 100000;
    float maxEdgeLen = 1.f/(powf(numTriangles,1.f/3.f));
    int numPoints    = 1000000;
    
    // the triangles we're querying to
    std::vector<vec3f> vertex;
    std::vector<vec3i> index;
    
    for (int i=0;i<numTriangles;i++) {
      const vec3f P(drand48(),drand48(),drand48());
      const vec3f e0 = maxEdgeLen * vec3f(drand48(),drand48(),drand48());
      const vec3f e1 = maxEdgeLen * vec3f(drand48(),drand48(),drand48());
      const int vIdx = vertex.size();
      vertex.push_back(P);
      vertex.push_back(P+e0);
      vertex.push_back(P+e1);
      index.push_back(vec3i(vIdx+0,vIdx+1,vIdx+2));
    }
    // loadTriangleMesh(vertex,index,av[1]);
    
    // the points we're query'ing
    std::vector<vec3f> queryPoint;
    //    loadQueryPoints(queryPoint,av[2]);
    for (int i=0;i<numPoints;i++) {
      queryPoint.push_back(vec3f(drand48(),drand48(),drand48()));
    }

    // the result of our queries
    std::vector<vec3f>   result_pos;    result_pos.resize(queryPoint.size());
    std::vector<int32_t> result_primID; result_primID.resize(queryPoint.size());
    std::vector<float>   result_dist;   result_dist.resize(queryPoint.size());

    auto begin = std::chrono::system_clock::now();
    // create the actual scene:
    distance_query_scene scene
      = rtdqNewTriangleMeshfi(&vertex[0].x,&vertex[0].y,&vertex[0].z,3,
                              &index[0].x,&index[0].y,&index[0].z,3,
                              index.size());
    auto done_build = std::chrono::system_clock::now();


    // perform the queries - all together, in a single thread
#if USE_TBB
    int blockSize = 1000;
    int numBlocks = (queryPoint.size()+blockSize-1)/blockSize;
    ::parallel_for(numBlocks, [&](size_t blockID){
        size_t begin = blockID*blockSize;
        size_t end = std::min(begin+blockSize,queryPoint.size());
        rtdqComputeClosestPoints(scene,
                                 &result_pos[begin].x,&result_pos[begin].y,&result_pos[begin].z,3,
                                 &result_dist[begin],1,
                                 &result_primID[begin],1,
                                 &queryPoint[begin].x,&queryPoint[begin].y,&queryPoint[begin].z,3,
                                 end-begin);
      });
#else
    rtdqComputeClosestPoints(scene,
                             &result_pos[0].x,&result_pos[0].y,&result_pos[0].z,3,
                             &result_dist[0],1,
                             &result_primID[0],1,
                             &queryPoint[0].x,&queryPoint[0].y,&queryPoint[0].z,3,
                             numPoints);
#endif
    auto done_all = std::chrono::system_clock::now();

    std::chrono::duration<double> buildTime = done_build - begin;
    std::chrono::duration<double> queryTime = done_all   - done_build;
    std::cout << "time to build tree " << buildTime.count() << "s" << std::endl;
    std::cout << "time to query " << numPoints << " points: " << queryTime.count() << "s" << std::endl;
    std::cout << "(this is " << (queryTime.count()/numPoints) << " seconds/prim)" << std::endl;

    rtdqDestroy(scene);
  }

}
