/*
  Modified by Alec Chen on Mar. 8th, 2019.

  Copyright (C) 2006 Pedro Felzenszwalb

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#ifndef SERVER_BASIC_SEGMENT_GRAPH_H_
#define SERVER_BASIC_SEGMENT_GRAPH_H_

#include <algorithm>
#include <cmath>
#include "disjoint_set.h"

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
 * nu_vertices: number of vertices in graph.
 * nu_edges: number of edges in graph
 * edges: array of edges.
 * c: constant for treshold function.
 */
universe *segment_graph(int nu_vertices, int nu_edges, edge *edges, float c) {
  // sort edges by weight
  std::sort(edges, edges + nu_edges);

  // make a disjoint-set forest
  universe *u = new universe(nu_vertices);

  // init thresholds
  float *threshold = new float[nu_vertices];
  for (int i = 0; i < nu_vertices; i++)
    threshold[i] = THRESHOLD(1,c);

  // for each edge, in non-decreasing weight order...
  for (int i = 0; i < nu_edges; i++) {
    edge *pedge = &edges[i];

    // components conected by this edge
    int a = u->find(pedge->a);
    int b = u->find(pedge->b);
    if (a != b) {
      if ((pedge->w <= threshold[a]) &&
          (pedge->w <= threshold[b])) {
        u->join(a, b);
        a = u->find(a);
        threshold[a] = pedge->w + THRESHOLD(u->size(a), c);
      }
    }
  }

  // free up
  delete threshold;
  return u;
}

#endif
