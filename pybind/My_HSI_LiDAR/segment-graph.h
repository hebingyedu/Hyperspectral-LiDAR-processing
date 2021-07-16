#ifndef SEGMENTGRAPH
#define SEGMENTGRAPH

#include <algorithm>
#include <cmath>
#include "disjoint-set.h"
#include <vector>
#include <iostream>

using namespace std;

// threshold function
#define THRESHOLD(size, c) (c/size)

typedef struct {
  float w;
  int a, b;
} edge;

bool operator<(const edge &a, const edge &b) {
  return a.w < b.w;
}

/*
 * Segment a graph
 *
 * Returns a disjoint-set forest representing the segmentation.
 *
 * num_vertices: number of vertices in graph.
 * num_edges: number of edges in graph
 * edges: array of edges.
 * c: constant for treshold function.
 */
universe *segment_graph(int num_vertices, int num_edges, std::vector<edge> & edges,
            float c) {
  // sort edges by weight

  std::sort(edges.begin() , edges.end());

  // make a disjoint-set forest
  universe *u = new universe(num_vertices);

  // init thresholds
  float *threshold = new float[num_vertices];
  for (int i = 0; i < num_vertices; i++)
    threshold[i] = THRESHOLD(1,c);
//cout<< "asd"<<endl;
  // for each edge, in non-decreasing weight order...
//cout<<edges.size()<<" "<<num_edges<<endl;
  for (int i = 0; i < num_edges; i++) {
    edge *pedge = &edges[i];

//cout<< " assd"<<endl;
    // components conected by this edge
    int a = u->find(pedge->a);
//    cout<< " assd1"<<endl;
    int b = u->find(pedge->b);
//    cout<< " assd2"<<endl;
    if (a != b) {
      if ((pedge->w <= threshold[a]) &&
      (pedge->w <= threshold[b])) {
         // cout<< " assd3"<<endl;
    u->join(a, b);
//    cout<< " assd4"<<endl;
    a = u->find(a);
//    cout<< " assd5"<<endl;
    threshold[a] = pedge->w + THRESHOLD(u->size(a), c);
      }
    }
  }

  // free up
  delete threshold;
  return u;
}


#endif // SEGMENTGRAPH

