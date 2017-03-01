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

#include "embree2/rtcore.h"
#include "embree2/rtcore_scene.h"

#include <vector>
#include <stdexcept>
#include <memory>
#include <chrono>

#include "../../kernels/bvh/bvh.h"
#include "../../kernels/geometry/object.h"

namespace embree {

  typedef Vec3<float> vec3f;
  typedef Vec3<int>   vec3i;

  /*! the embree callback to compute a primitive's bounds; we do all
    floats right now, even if the object internally uses doubles */
  void boundsCallback(void* ptr, size_t primID, RTCBounds& bounds);

  /*! the embree callback to compute a ray-prim intersection - since
    we never actually trace any rays (we only use the BVH buidler)
    we don't actually need this */
  void isecCallback(void* ptr, RTCRay& ray, size_t primID);



  /*! result of a single closest-point query */
  struct QueryResult {
    
    /*! closest point on given triangle */
    vec3f   point;
    
    /*! distance to query point */
    float   distance;
    
    /* primitmive it in scene object */
    int32_t primID;
  };

  /*! a single trianlge's worth of vertices, so we can isolate the
      algorithms that operate on triangles from the actual data
      format */
  struct Triangle {
    inline Triangle(const vec3f &A, const vec3f B, const vec3f C)
      : A(A), B(B), C(C)
    {}

    const vec3f A;
    const vec3f B;
    const vec3f C;
  };

  inline float clamp01(const float a, float b)
  {
    if (embree::abs(b) < 1e-12f) b = 1e-12f;
    float f = a / b;
    return min(max(f,0.f),1.f);
  }

  inline vec3f projectToEdge(const vec3f P, const vec3f A, const vec3f B)
  {
    float f = dot(P-A,B-A) / dot(B-A,B-A);
    f = max(0.f,min(1.f,f));
    return A+f*(B-A);
  }

  inline vec3f projectToPlane(const vec3f P, const vec3f N, const vec3f A)
  {
    return P - dot(P-A,N) * N;
    // float f = dot(P-A,B-A) / dot(B-A,B-A);
    // f = max(0.f,min(1.f,f));
    // return A+f*(B-A);
  }

  inline void checkEdge(vec3f &closestPoint, float &closestDist,
                        const vec3f &queryPoint,
                        const vec3f &v0, const vec3f &v1)
  {
    const vec3f PP = projectToEdge(queryPoint,v0,v1);
    const float dist = length(PP-queryPoint);
    if (dist < closestDist) {
      closestDist = dist;
      closestPoint = PP;
    }
  }

  /*! compute the closest point P' to P on triangle ABC, and return it */
  inline vec3f closestPoint(const Triangle &triangle,
                            const vec3f &QP,
                            size_t primID)
  {
    const vec3f N = normalize(cross(triangle.B-triangle.A,triangle.C-triangle.A));
    const vec3f Na = normalize(cross(N,triangle.C-triangle.B));
    const vec3f Nb = normalize(cross(N,triangle.A-triangle.C));
    const vec3f Nc = normalize(cross(N,triangle.B-triangle.A));
    
    float a = dot(QP-triangle.B,Na);// / dot(A-B,Na);
    float b = dot(QP-triangle.C,Nb);// / dot(B-C,Nb);
    float c = dot(QP-triangle.A,Nc);// / dot(C-A,Nc);

    vec3f closest;
    if (min(min(a,b),c) >= 0.f)
      closest = projectToPlane(QP,N,triangle.A);
    else {
      float closestDist = std::numeric_limits<float>::infinity();
      if (a <= 0.f) 
        checkEdge(closest,closestDist,QP,triangle.B,triangle.C);
      if (b <= 0.f) 
        checkEdge(closest,closestDist,QP,triangle.C,triangle.A);
      if (c <= 0.f) 
        checkEdge(closest,closestDist,QP,triangle.A,triangle.B);
    }
    
    return closest;
  }

  struct QueryObject {
    virtual ~QueryObject() {};
    virtual size_t size() const = 0;
    virtual void computeBounds(RTCBounds &bounds, const size_t primID) = 0;
    virtual void updateIfCloser(QueryResult &resut,
                                const vec3f &P,
                                const size_t primID) = 0;
    RTCScene  scene  { 0 };
    RTCDevice device { 0 };
    int       geomID { 0 };
  };

  template<typename coord_t, typename index_t>
  struct GeneralTriangleMesh : public QueryObject {

    /*! constructor */
    GeneralTriangleMesh(const coord_t *vtx_x,
                        const coord_t *vtx_y,
                        const coord_t *vtx_z,
                        const size_t   vtx_stride,
                        const index_t *idx_x,
                        const index_t *idx_y,
                        const index_t *idx_z,
                        const size_t   idx_stride,
                        const size_t   numTriangles);

    /*! destructor - release the embree scene and device */
    virtual ~GeneralTriangleMesh();

    /*! number of primitmives in this query object (mostly for
      debugging) */
    virtual size_t size() const { return numTriangles; }

    /*! get the vertices for one triangle */
    inline Triangle getTriangle(const size_t primID);

    /*! compute bounding box of given primtimive (returned in float,
      whatever the input data is */
    virtual void computeBounds(RTCBounds &bounds, const size_t primID);

    /*! test one candidate primitmive, and update the currently
      closest point (in 'results') if it is closer */
    virtual void updateIfCloser(QueryResult &result,
                                const vec3f &P,
                                const size_t primID);

    const coord_t *const vtx_x;
    const coord_t *const vtx_y;
    const coord_t *const vtx_z;
    const size_t   vtx_stride;
    const index_t *const idx_x;
    const index_t *const idx_y;
    const index_t *const idx_z;
    const size_t   idx_stride;
    const size_t   numTriangles;

  };

  /*! constructor */
  template<typename coord_t, typename index_t>
  GeneralTriangleMesh<coord_t,index_t>::GeneralTriangleMesh(const coord_t *vtx_x,
                                                            const coord_t *vtx_y,
                                                            const coord_t *vtx_z,
                                                            const size_t   vtx_stride,
                                                            const index_t *idx_x,
                                                            const index_t *idx_y,
                                                            const index_t *idx_z,
                                                            const size_t   idx_stride,
                                                            const size_t   numTriangles)
    : vtx_x(vtx_x),
      vtx_y(vtx_y),
      vtx_z(vtx_z),
      vtx_stride(vtx_stride),
      idx_x(idx_x),
      idx_y(idx_y),
      idx_z(idx_z),
      idx_stride(idx_stride),
      numTriangles(numTriangles)
  {
    this->device = rtcNewDevice("object_accel=bvh4.object");
    this->scene   = rtcDeviceNewScene(device,RTC_SCENE_STATIC,RTC_INTERSECT1);
    this->geomID  = rtcNewUserGeometry(scene, numTriangles);
    rtcSetUserData(scene,geomID,this);
    rtcSetBoundsFunction(scene,geomID,boundsCallback);
    rtcSetIntersectFunction(scene,geomID,isecCallback);
    rtcCommit(scene);
  }

  /*! destructor - release the embree scene and device */
  template<typename coord_t, typename index_t>
  GeneralTriangleMesh<coord_t,index_t>::~GeneralTriangleMesh()
  {
    rtcDeleteScene(scene);
    rtcDeleteDevice(device);
  }

  /*! get the vertices for one triangle */
  template<typename coord_t, typename index_t>
  Triangle GeneralTriangleMesh<coord_t,index_t>::getTriangle(const size_t primID)
  {
    const size_t i0 = this->idx_x[primID*idx_stride];
    const size_t i1 = this->idx_y[primID*idx_stride];
    const size_t i2 = this->idx_z[primID*idx_stride];
    
    const vec3f v0(this->vtx_x[i0*vtx_stride],
                   this->vtx_y[i0*vtx_stride],
                   this->vtx_z[i0*vtx_stride]);
    const vec3f v1(this->vtx_x[i1*vtx_stride],
                   this->vtx_y[i1*vtx_stride],
                   this->vtx_z[i1*vtx_stride]);
    const vec3f v2(this->vtx_x[i2*vtx_stride],
                   this->vtx_y[i2*vtx_stride],
                   this->vtx_z[i2*vtx_stride]); 
    return Triangle(v0,v1,v2);
  }
    
  /*! compute bounding box of given primtimive (returned in float,
    whatever the input data is */
  template<typename coord_t, typename index_t>
  void GeneralTriangleMesh<coord_t,index_t>::computeBounds(RTCBounds &bounds, const size_t primID)
  {
    const Triangle triangle = getTriangle(primID);
    const vec3f lo = min(min(triangle.A,triangle.B),triangle.C);
    const vec3f hi = max(max(triangle.A,triangle.B),triangle.C);
    (vec3f &)bounds.lower_x = lo;
    (vec3f &)bounds.upper_x = hi;
  }

  /*! test one candidate primitmive, and update the currently
    closest point (in 'results') if it is closer */
  template<typename coord_t, typename index_t>
  void GeneralTriangleMesh<coord_t,index_t>::updateIfCloser(QueryResult &result,
                                                            const vec3f &P,
                                                            const size_t primID)
  {
    const Triangle triangle = getTriangle(primID);
    const vec3f PP   = closestPoint(triangle,P,primID);
    const float dist = length(PP-P);
    if (dist >= result.distance) return;

    result.distance = dist;
    result.point    = PP;
    result.primID   = primID;
  }


  /*! the embree callback to compute a primitive's bounds; we do all
    floats right now, even if the object internally uses doubles */
  void boundsCallback(void* ptr, size_t primID, RTCBounds& bounds)
  {
    QueryObject *qo = (QueryObject *)ptr;
    qo->computeBounds(bounds,primID);
  }

  /*! the embree callback to compute a ray-prim intersection - since
    we never actually trace any rays (we only use the BVH buidler)
    we don't actually need this */
  void isecCallback(void* ptr, RTCRay& ray, size_t primID)
  {
    throw std::runtime_error("this should never get called for distance queries");
  }

  extern "C"
  distance_query_scene rtdqNewTriangleMeshfi(const float   *vertex_x,
                                             const float   *vertex_y,
                                             const float   *vertex_z,
                                             const size_t   vertex_stride,
                                             const int32_t *index_x,
                                             const int32_t *index_y,
                                             const int32_t *index_z,
                                             const size_t   index_stride,
                                             const size_t   numTriangles)
  {
    return (distance_query_scene)
      new GeneralTriangleMesh<float,int>(vertex_x,vertex_y,vertex_z,vertex_stride,
                                         index_x,index_y,index_z,index_stride,
                                         numTriangles);
  }
  
  extern "C"
  distance_query_scene rtdqNewTriangleMeshdi(const double   *vertex_x,
                                             const double   *vertex_y,
                                             const double   *vertex_z,
                                             const size_t   vertex_stride,
                                             const int32_t *index_x,
                                             const int32_t *index_y,
                                             const int32_t *index_z,
                                             const size_t   index_stride,
                                             const size_t   numTriangles)
  {
    return (distance_query_scene)
      new GeneralTriangleMesh<double,int>(vertex_x,vertex_y,vertex_z,vertex_stride,
                                          index_x,index_y,index_z,index_stride,
                                          numTriangles);
  }

  inline vfloat<4> computeDistance(const BVH4::AlignedNode *node, const vec3f &P)
  {
    vfloat<4> clamped_x = min(max(P.x,node->lower_x),node->upper_x);
    vfloat<4> clamped_y = min(max(P.y,node->lower_y),node->upper_y);
    vfloat<4> clamped_z = min(max(P.z,node->lower_z),node->upper_z);

    vfloat<4> dx = clamped_x - P.x;
    vfloat<4> dy = clamped_y - P.y;
    vfloat<4> dz = clamped_z - P.z;

    return dx*dx + dy*dy + dz*dz;
  }
  
  void oneQuery(QueryResult &result,
                QueryObject *qo,
                const vec3f &point)
  {
    result.distance = std::numeric_limits<float>::infinity();
    result.primID   = -1;
  
    // we already type-checked this before tihs fct ever got called,
    // so this is safe:
    BVH4 *bvh4 = (BVH4*)((Accel*)qo->scene)->intersectors.ptr;

    std::priority_queue<std::pair<float,BVH4::NodeRef>,
      std::vector<std::pair<float,BVH4::NodeRef>>,
      std::greater<std::pair<float,BVH4::NodeRef>>
      > queue;
    BVH4::NodeRef node = bvh4->root;
    while (1) {
      if (node.isAlignedNode()) {
        // this is a inner node ...
        BVH4::AlignedNode* n = node.alignedNode();
        vfloat<4> dist       = computeDistance(n,point);
        for (int i=0;i<4;i++)
          if (n->child(i) == BVH4::emptyNode) dist[i] = std::numeric_limits<float>::infinity();
        for (int i=0;i<4;i++)
          queue.push(std::pair<float,BVH4::NodeRef>(dist[i],n->child(i)));
      } else {
        /// this is a leaf node
        size_t numPrimsInLeaf = 0;
        embree::Object *leaf = (embree::Object*)node.leaf(numPrimsInLeaf);
        for (size_t primNum=0;primNum<numPrimsInLeaf;primNum++)
          qo->updateIfCloser(result,point,leaf[primNum].primID);
      }

      // any more candidates in queue?
      if (queue.empty()) break;

      // closest candidate is already too far?
      if (queue.top().first >= result.distance) break;

      // closest candidate might be closer: pop it and use it
      node = queue.top().second;
      queue.pop();
    }
  }
  
  extern "C"
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
                                const size_t numQueryPoints)
  {
    QueryObject *qo = (QueryObject *)scene;
    if (!qo)
      return;
    AccelData *accel = ((Accel*)qo->scene)->intersectors.ptr;
    if (!accel)
      return;
    if (accel->type != AccelData::TY_BVH4)
      return;

    for (size_t i=0;i<numQueryPoints;i++) {
      QueryResult qr;
      
      oneQuery(qr,qo,vec3f(in_query_point_x[i*in_query_point_stride],
                           in_query_point_y[i*in_query_point_stride],
                           in_query_point_z[i*in_query_point_stride]));
      if (out_closest_point_pos_x)
        out_closest_point_pos_x[i*out_closest_point_pos_stride] = qr.point.x;
      if (out_closest_point_pos_y)
        out_closest_point_pos_y[i*out_closest_point_pos_stride] = qr.point.y;
      if (out_closest_point_pos_z)
        out_closest_point_pos_z[i*out_closest_point_pos_stride] = qr.point.z;
      if (out_closest_point_primID)
        out_closest_point_primID[i*out_closest_point_primID_stride] = qr.primID;
      if (out_closest_point_dist)
        out_closest_point_dist[i*out_closest_point_dist_stride] = qr.distance;
    }
  }
  
  
  /*! destroy a scene created with rtdqNew...() */
  extern "C"
  void rtdqDestroy(distance_query_scene scene)
  {
    if (scene) delete (QueryObject*)scene;
  }

  
}
 
