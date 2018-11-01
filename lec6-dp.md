# lec6-dp
## 有向无环图最短路径
```
initialize all dist to INF
dist(s) = 0
for each v in V except s in linearized order:
  dist(v) = min(u, v) in E{ dist(u) + l(u, v) }
```
## 最长递增子序列
### 一般算法
```
for j in n:
  L(j) = 1 + max{ L(i):(i, j) in E}
return max L(j)
```
### 利用二分查找减小复杂度
B是一个数组，B(i)代表长度为i的递增序列的末尾最小值。初始化为无穷
对于每个在数组中的元素：
  找到这个元素应该放入B的区间位置。如：B(1) = 2, B(2) = 3, B(3) = 9 ...(INF)，拿到元素4时，显然，3<4<9，因此将4放入到3号下标中。查找区间的方法用二分查找。
## 背包
### (0-1) 背包：N件物品，第i件物品的体积是c(i)，价值是w(i)，背包容量为V
f(i)(v)表示前i件放入一个容量为v的别抱可以获得的最大价值。f(N)(V)就是答案。假设对于只有i-1件物品的情况，我们求得背包容量从0-V时的最优值，也就是说，所有的f(i-1)(v)都是已知的。那么，状态转移方程为：<br>
`f(i)(v) = max(f(i-1)(v), f(i-1)(v-c(i))+w(i)), v >= c(i)`<br>
`f(i)(v) = f(i-1)(v)`<br>
时间复杂度为：O(NV)；空间复杂度：O(NV)
```
for v = 0 to V:
  f(0)(v) = 0
for i = 0 to N:
  for v = 0 to V:
    if v < c(i): f(i)(v) = f(i-1)(v)
    else: f(i)(v) = max(f(i-1)(v), f(i-1)(v-c(i)) + w(i))
```
节省空间复杂度：
```
for v = 0 to V: f(v) = 0
for i = 1 to N:
  for v = V to c(i):
    f(v) = max(f(v), f(v-c(i)) + w(i))
```
### 可重复背包
每种物品都有无限件可用，第i种物品的费用是c(i)，价值是w(i)。状态转移方程：<br>
`f(v) = max(f(v-c(i)) + w(i))`
```
for v = 1 to V:
  for i = 1 to N:
    if v >= c(i): f(v) = max(f(v-c(i)) + w(i), f(v))
```
### 二维背包
对于每件问题，具有两种不同的费用；选择这件物品需要同时付出两种代价；对于每种代价都有一个可付出的最大值（背包容量）。求出最大价值。代价:a[i] b[i]; 两种代价最大值: V U。物品价值: w[i]<br>
状态转移：`f[i][v][u] = max { f[i-1][v][u], f[i-1][v-a[i]][u-b[i]]+w[i] }`<br>

### 分组背包
有N件物品，容量为V的背包，第i件物品的费用是c[i]，价值是w[i]。这些物品被划分为若干组，每组中的物品相互冲突，最多选一件。求最大价值。<br>
f[k][v]表示前k组物品花费费用v能获得的最大值。<br>
`f[k][v] = max { f[k-1][v], f[k-1][v-c[i]]+w[i] | 物品属于组k }`<br>

## 编辑距离问题
输入两个字符串x和y，修改x到y需要的最小操作步数叫编辑距离（插入，删除、替换）。<br>
```
SNOWY->SUNNY:
SNOWY -> SUNOWY -> SUNNWY -> SUNNY
编辑距离为：3
```
设两个字符串分别为x[1...m], y[1...n]，用E(i, j)表示x[1...i]和y[1...j]的编辑距离，则可以得到动态规划转移方程：<br>
```
当i=0, E(i, j) = j;
当j=0, E(i, j) = i;
当i>0且j大于0时：
E(i, j) = min{1+E(i-1, j), 1+E(i, j-1), diff(i, j)+E(i-1, j-1)}
其中，当x[i]=y[j]时，diff(i, j)=0; 否则diff(i, j)=1
```

## 矩阵连乘
(A1A2A3) <=> ((A1A2)A3) <=> (A1(A2A3))，两种不同的顺序导致乘法次数不一样。<br>
给定n个矩阵构成的链<A1...An>, 其中矩阵Ai的大小为p[i-1] * p[i], i = 1..n，找一种计算顺序，使得计算乘积A1A2...An的乘法次数最少。<br>
令m[i,j]表示计算Ai...Aj的最小乘法次数，得到递归方程：<br>
`m[i,j]=min{m[i,k]+m[k+1,j]+pi-1*pk*pj} i < j; i==j: m[i,j]=0`
```
MATRIX-CHAIN-ORDER(p)
n = length(p) - 1
for i from 1 to n: m[i, i] = 0
for l from 2 to n:
  for i from 1 to n-l+1
    j = i + l - 1
      m[i, j] = INF
        for k from i to j-1
         q = m[i, k] + m[k+1, j] + pi-1*pk*pj
          if q < m[i, j] then m[i, j] = q, s[i, j] = k
return m and s
```
## 最短可靠路径
给定一个带权图G，图中的两个节点s和t，以及正整数k，求从s到t的不超过k条边的最短路。<br>
dist[v, i]表示从起点s到v点经过i条边的最短路。<br>
`dist[v, i] = min { dist[u, i-1] + l(u, v) }`

## 最短路问题分类
+ 单源最短路
+ 单点对单点最短路
+ 所有点对最短路

## Floyd-warshall算法
dist(i,j,k): 从i到j的仅使用节点（1-k）作为中间节点的最短路的长度。<br>
转移方程:`dist(i,j,k)=min{dist(i,j,k-1), dist(i,k,k-1)+dist(k,j,k-1)}`<br>
是否经过最后一个点k。
```
for i = 1 to n
  for j = 1 to n
    dist(i,j,0) = INF
for all (i,j) in E
  dist(i,j,0) = INF
// k作为外层循环保证内层循环中dist(i,j,k-1)已经被算出来
for k = 1 to n
  for i = 1 to n
    for j = 1 to n
      dist(i,j,k) = min{dist(i,k,k-1) + dist(k,j,k-1), dist(i,j,k-1)}
```
## 状态压缩DP
对于集合S={0~N-1}, 在n比较小的情况下，S的子集A可以用一个N位的二进制数表示：当i属于A时，x的第i位为1；否则为0.<br>
当n=16时，A={2，3，5} = 001101
### 图的最长路
给出一个带权有向图G=(V,E)，求出图中的一条最长简单路径，所谓简单路径是指路径中的节点不重复。<br>
暴力穷举：O(N!)<br>
状态表示：用ans[j][i]表示以j为起点，并且经过点集i中的点恰好一次且不经过其他点的路径长度的最大值。如果这个状态不存在，就是无穷小。<br>
状态转移：如果i只包含一个点，ans[j][i]=0;<br>
否则，`ans[j][i] = max(G(j, k) + ans[k][s])`s表示i集合中去掉了j点的集合，k遍历集合s中的点，点j到点k有边存在，G(j,k)表示边(j,k)的权值。

## 树型DP
在树结构上的动态规划。树本身就是一种递归的结构。很多在一般图结构上是NP难得傅问题，在树结构上存在多项式时间的DP算法
+ 最小点覆盖问题
+ 最大独立值问题
+ etc
算法实现通常借助DFS过程。
### 树的最小点覆盖
给出一个n个节点的树，要求选出其中的一些顶点，使得对于树中的每条边(u,v)，u和v至少一个被选中。<br>
在树结构中，每个节点都是某棵子树的根。<br>
给0~n-1编号，用ans[i][0]表示：在不选择i节点的情况下，以i为根党的子树，最少需要选择的点数；用ans[i][1]表示选择节点i的情况下，以i为根节点的子树最少需要选择的点数。<br>
当i是叶子时，ans[i][0] = 0,ans[i][1] = 1，否则`ans[i][0] = sum(ans[j][i])`（对于i的所有子节点j），`ans[i][1] = 1 + sum(min{ans[j][0], ans[j][1]})`