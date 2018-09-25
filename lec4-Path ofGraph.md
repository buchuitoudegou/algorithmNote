# lec4 图中的路径
## BFS
DFS能找到从起点到其他点的路径，但不能保证找到最短的。
```
dist(s) = 0
Q = [s]
while Q is not empty
  u = eject(Q)
  for all edges(u, v) in E
    if dist(v) = INF
      inject(Q, v)
      dist(v) = dist(s) + 1
```

#### 复杂度：O(V+E)

### 隐式图的BFS
问题中没有给出明确的图结构，图需要一边做BFS一边构造

#### 8数码问题
标有1-8的八块正方形数码拍任意地放在3x3的数码盘上。每次只能将与空格相邻的数码牌与空格交换。将任意摆放的数码盘以最少的步数摆成某种特殊的排列。<br>

将排列状态当成一个节点。

### 带权图的单源最短路
#### 简单想法一
将长度换成节点，然后对新图进行BFS。当长度很大时，插入的节点太多，不实际。
#### 想法二Alarm clocks
每秒走一步，走到节点时响应。
#### Dijkstra算法
```
for all u in V
  dist(u) = INF
  prev(u) = nil
dist(s) = 0
H = makequeue(V) // dist-value as keys优先队列
while H is not empty
  u = deletemin(H)
  for all edges (u, v) in E
    if dist(v) > dist(u) + l(u, v)
      dist(v) = dist(u) + l(u, v)
      prev(v) = u
      decreaseKey(H, v) // 更新到源点的距离值
```