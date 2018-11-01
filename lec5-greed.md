# lec5 - greed algorithm
## prim算法
for all v in V
  cost(v) = INF
  prev(v) = null

initialize v0
cost(v0) = 0

H = priorityQueue(V)
while H is not empty
  v = deleteMin(H)
  for each { v, z } in E
    if cost(z) > w(v, z)
      cost(z) = w(v, z)
      prev(z) = v
      decreaseKey(H, z)

## kruskal算法
for all u in V
  makeset(u) // initial Union-find Set
X = {}
sort edges by weight
for all edges{ u, v } in E:
  if find(u) != find(v) // not the same parent
    add edge{u, v} to X
    union(u, v)// set to the same parent

## huffman
