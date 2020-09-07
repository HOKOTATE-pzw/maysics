import numpy as np


def ek(G, ori, des):
    '''
    EK算法
    利用BFS遍历求解最大流问题
    
    参数
    ----
    G：邻接矩阵(非邻接点间的边权重(即流量)为inf或负数或0)
    ori：整型，起点
    des：整型，终点
    
    
    Edmond-Karp Algorithm
    Solve maximum flow problem by BFS
    
    Parameters
    ----------
    G: adjacency matrix(the weight(flow) of the edge between non-adjacent nodes is inf, negative or 0)
    ori: int, origin
    des: int, destination
    '''
    G = np.array(G, dtype=np.float)
    n = len(G)
    G[G < 0] = 0
    G[G == float('inf')] = 0
    max_flow = 0
    
    while True:
        visited = np.zeros(n)
        qu = [[ori, -1]]
        path = []
        visited[ori] = 1
        
        i = 0
        judge = False
        while i < len(qu):
            for j in range(n):
                if visited[j] == 0 and G[qu[i][0]][j] > 0:
                    visited[j] = 1
                    qu.append([j, i])
                
                    if j == des:            #若找到简单路径则break
                        judge = True
                        break
            
            if judge:
                break
            
            if i == len(qu)-1:              #若没有简单路径则程序结束
                return max_flow
            
            i += 1
        
        path.append(qu[-1][0])
        loc = qu[-1][1]
        while loc != -1:
            path.append(qu[loc][0])
            loc = qu[loc][1]
        
        dis = []                            #将path中各边流量存入dis
        for i in range(len(path)-1):
            dis.append(G[path[i+1], path[i]])
        
        mindis = min(dis)
        max_flow += mindis                  #更新最大流
        
        for i in range(len(path)-1):        #更新邻接矩阵
            G[path[i+1], path[i]] -= mindis
            G[path[i], path[i+1]] += mindis


def Kruskal(G):
    '''
    克鲁斯卡尔算法
    生成最小生成树
    
    参数
    ----
    G：图的邻接矩阵(非邻接点间的边权重为inf或负数)
    
    返回值：列表，(权值，起点，终点)
    
    
    Kruskal Algorithm
    Generate minimum spanning tree
    
    Parameter
    ---------
    G: adjacency matrix(the weight of the edge between non-adjacent nodes is inf or negative)
    
    return: list, (weight, origin, destination)
    '''
    G = np.array(G, dtype=np.float)
    n = len(G)
    G[G < 0] = float('inf')
    E = []
    
    # 将边信息(边长，起点，终点)存入self.E并按从小到大排序
    for i in range(n):
        for j in range(i):
            if G[i, j] != 0 and G[i, j] != float('inf'):
                E.append([G[i, j], i, j])
    E.sort()
        
    path = []
    vset = np.arange(n)
    k = 1
    j = 0
    while k < n:
        ul = E[j][1]        #第j条边的起点
        vl = E[j][2]        #第j条边的终点
        sn1 = vset[ul]
        sn2 = vset[vl]
        if sn1 != sn2:
            path.append(E[j])
            k += 1
            vset[vset == sn2] = sn1
        j += 1
    
    return path



class Floyd():
    '''
    弗洛伊德算法
    
    参数
    ----
    G：图的邻接矩阵(非邻接点间的边权重为inf或负数)
    
    属性
    ----
    A：二维列表，A[i, j]代表节点i到节点j的距离
    
    
    Floyd Algorithm
    
    Parameter
    ---------
    G: adjacency matrix(the weight of the edge between non-adjacent nodes is inf or negative)
    
    Attribute
    ---------
    A: 2-D list, A[i, j] is the distance from node i to node j
    '''
    def __init__(self, G):
        self.A = np.array(G, dtype=np.float)
        n = len(self.A)
        self.A[self.A < 0] = float('inf')
        for i in range(n):
            self.A[i, i] = 0
        
        self.__path = np.zeros((n, n), dtype=np.int)
        for i in range(n):
            for j in range(n):
                if i != j:
                    if 0 <= self.A[i, j] < float('inf'):
                        self.__path[i, j] = i
                    else:
                        self.__path[i, j] = -1
                else:
                    self.__path[i, j] = -1
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if self.A[i, j] > self.A[i, k] + self.A[k, j]:
                        self.A[i, j] = self.A[i, k] + self.A[k, j]
                        self.__path[i, j] = self.__path[k, j]
    
    
    def path(self, ori, des):
        '''
        求解两节点之间的最短路径
        
        参数
        ----
        ori：整型，起点
        des：整型，终点
        
        
        Find the shortest path between two nodes
        
        Parameters
        ----------
        ori: int, origin
        des: int, destination
        '''
        if self.A[ori, des] == float('inf'):
            raise Exception('Unable to reach the destination from the origin.')
        
        elif ori == des:
            raise Exception('The origin and the destination should be different.' )
        
        else:
            distance = self.A[ori, des]
            
            path = [des]
            while path[-1] != ori:
                path.append(self.__path[ori, des])
                des = self.__path[ori, des]
            path.reverse()
            
            return distance, path
    
    
    def circle(self, point):
        '''
        求解某一节点的最短有向环
        
        参数
        ----
        point：整型或浮点数类型，节点
        
        
        Find the shortest directed circle of a node
        
        Parameter
        ---------
        point: int or float, node
        '''
        circle_distance_list = self.A[point] + self.A[:, point]
        circle_distance_list[point] = float('inf')
        target = np.argmin(circle_distance_list)
        distance = np.min(circle_distance_list)
        if distance == float('inf'):
            raise Exception("There is no proper circle.")
        
        target_1 = target
        path1 = [target_1]
        while path1[-1] != point:
            path1.append(self.__path[point, target_1])
            target_1 = self.__path[point, target_1]
        path1.reverse()
        
        target_2 = point
        path2 = [target_2]
        while path2[-1] != target:
            path2.append(self.__path[target, target_2])
            target_2 = self.__path[target, target_2]
        path2.reverse()
        path2.remove(path2[0])
        
        return distance, path1+path2


class Dijkstra():
    '''
    狄克斯特拉算法
    
    参数
    ----
    G：图的邻接矩阵(非邻接点间的边权重为inf或负数)
    ori：整型，起点
    
    属性
    ----
    dis：一维列表，dis[i]表示起点到节点i的距离
    
    
    Dijkstra Algorithm
    
    Parameters
    ----------
    G: adjacency matrix(the weight of the edge between non-adjacent nodes is inf or negative)
    ori: int, origin
    
    Attribute
    ---------
    dis: 1-D list, dis[i] is the distance from the origin to node i
    '''
    def __init__(self, G, ori):
        self.__ori = ori
        G = np.array(G, dtype=np.float)
        G[G < 0] = float('inf')
        n = len(G)
        self.dis = G[ori].copy()
        S = np.zeros(n)
        self.__path = self.dis.copy()
        self.__path[self.__path < float('inf')] = ori
        self.__path[self.__path == float('inf')] = -1
        self.__path = np.array(self.__path, dtype=np.int)
        
        S[ori] = 1
        self.__path[ori] = 0
        for i in range(n-1):
            mindis = float('inf')
            for j in range(n):
                if S[j] == 0 and self.dis[j] < mindis:
                    u = j
                    mindis = self.dis[j]
            S[u] = 1
            for j in range(n):
                if S[j] == 0:
                    if G[u, j] < float('inf') and self.dis[u] + G[u, j] < self.dis[j]:
                        self.dis[j] = self.dis[u] + G[u, j]
                        self.__path[j] = u
    
    
    def path(self, des):
        '''
        求解ori和des之间的最短路径
        
        参数
        ----
        des：整型，终点
        
        
        Find the shortest path from ori to des
        
        Parameter
        ---------
        des: int, destination
        '''
        if self.__path[des] == -1:
            raise Exception('Unable to reach the destination from the origin.')
        
        elif self.__ori == des:
            raise Exception('The origin and the destination should be different.' )
        
        else:
            distance = self.dis[des]
            
            path = [des]
            while path[-1] != self.__ori:
                path.append(self.__path[des])
                des = self.__path[des]
            path.reverse()
            
            return distance, path