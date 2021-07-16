#ifndef DISJOINTSET
#define DISJOINTSET

// disjoint-set forests using union-by-rank and path compression (sort of).

typedef struct {
  int rank;
  int p;
//  int p1;
  int size;
//  int bou;
//  int seg;
//  int boundin2;
} uni_elt;

class universe {
public:
  universe(int elements);
  ~universe();
  int find(int x);
  void join(int x, int y);
  int size(int x) const { return elts[x].size; }
  int num_sets() const { return num; }

public:
  uni_elt *elts;
  int num;
};

universe::universe(int elements) {
  elts = new uni_elt[elements];
  num = elements;
  for (int i = 0; i < elements; i++) {
    elts[i].rank = 0;
    elts[i].size = 1;
    elts[i].p = i;
//    elts[i].p1= -1;
//     elts[i].bou = 0;
//     elts[i].seg = 0;
//     elts[i].boundin2 = 0;
  }
}

universe::~universe() {
  delete [] elts;
}

int universe::find(int x) {
  int y = x;
  while (y != elts[y].p)
    y = elts[y].p;
  elts[x].p = y;
  return y;
}

void universe::join(int x, int y) {
  if (elts[x].rank > elts[y].rank) {
    elts[y].p = x;
    elts[x].size += elts[y].size;
  } else {
    elts[x].p = y;
    elts[y].size += elts[x].size;
    if (elts[x].rank == elts[y].rank)
      elts[y].rank++;
  }
  num--;
}


#endif // DISJOINTSET

