import numpy as np
import math
from matplotlib import pyplot as plt
import sys
import os 
from time import sleep
from PIL import Image

from Point import Point

class RRT():
    def __init__(self, delta_q, p, max_depth, min_dist):
        self.delta_q = delta_q
        self.p = p
        self.max_depth = max_depth
        self.min_dist = min_dist

    def rand_conf(self, C, p, qgoal):
        """Random the configuration qrand in C with a bias to the qgoal \n
        Args:
        - C: robot configuration
        - p: bias probability of q_rand to be qgoal
        - qgoal: goal position"""
        if np.random.random() < p:
            return qgoal
        # else:
            # qrand = []
            # conf_shape = C.shape[0]
            # get_qrand = False 
            # while(get_qrand == False):
            #     qrand = np.random.randint(0, conf_shape, 2)
            #     if C[qrand[0], qrand[1]] == 0:
            #         get_qrand = True
            # qrand = Point(qrand[0], qrand[1])
        return np.random.choice(self.c_free_space)

    def nearest_vertex(self, qrand, G):
        """Returns the vertice in graph G that has the smallest Eucledian distance to qrand \n
        Args:
        - qrand: random q
        - G: list of vertices in the tree"""
        qnear = G[0]
        nearest_dist = np.inf
        for i, vertice in enumerate(G):
            # q_x , q_y = vertice
            # dist = np.sqrt(np.square(q_x - qrand[0]) + np.square(q_y - qrand[1]))
            dist = vertice.dist(qrand)
            if dist < nearest_dist:
                qnear = vertice
                nearest_dist = dist
        return qnear

    def new_conf(self, q_near, q_rand):
        """Returns a new configuration qnew by moving an incremental distance delta_q from qnear  in the direction of  qrand  without overshooting  qrand \n
        Agrs:
        - qnear: nearest q in G to qrand
        - qrand: random configuration q
        - delta_q: incremental distance"""
        q_near_vector = q_near.vector(q_rand)
        if q_near_vector.norm() < delta_q:
            return q_rand
        q_near_vector_unit = q_near_vector.unit()
        q_new = Point(q_near.x + delta_q * q_near_vector_unit.x,
                      q_near.y + delta_q * q_near_vector_unit.y)
        q_new = Point(round(q_new.x), round(q_new.y))
        return q_new

    def is_point_occupied(self, q, C):
        if q.x < 0 or q.x > C.shape[0] - 1 or q.y < 0 or q.y > C.shape[1] - 1:
            return True
        if int(q.x) >= C.shape[0] - 1 or int(q.y) >= C.shape[1] - 1:
                return C[int(q.x)][int(q.y)] == 1
        x = int(q.x)
        y = int(q.y)
        return C[x, y] == 1 or C[x+1, y] == 1 or C[x, y+1] == 1 or C[x+1, y+1] == 1 # round up error compensation

    def is_segment_free_bisection(self, qnear, qnew, C, depth):
        # Recursive function to check two-half segments free or occupied
        if qnear.x == qnew.x and  qnear.y == qnew.y:
            return False
        
        if self.is_point_occupied(qnear, C):
            return False
        if self.is_point_occupied(qnew, C):
            return False
        
        mid_point = qnear.__add__(qnew).__truediv__(2)
        if depth >= self.max_depth:
            return True
        
        if self.is_point_occupied(mid_point, C):
            return False
        
        left_segment = self.is_segment_free_bisection(qnear, mid_point, C, depth+1)
        if left_segment == False:
            return False
        right_segment = self.is_segment_free_bisection(mid_point, qnew, C, depth+1)
        return right_segment

    def sample(self, C, K, p, qstart_x, qstart_y, qgoal_x, qgoal_y):
        qstart = Point(qstart_x, qstart_y)
        qgoal = Point(qgoal_x, qgoal_y)
        G = [qstart]
        edges = []

        self.c_free_space = []
        for i, j in np.ndindex(C.shape):
            if C[i][j] == 0:
                self.c_free_space.append(Point(i, j))

        for k in range(K):
            qrand = self.rand_conf(C, p, qgoal)
            qnear = self.nearest_vertex(qrand, G)
            qnew = self.new_conf(qnear, qrand)
            if self.is_segment_free_bisection(qnear, qnew, C, 0):
                G.append(qnew)
                edges.append([G.index(qnear), len(G)-1])
                dist = qnew.dist(qgoal)
                if dist < self.min_dist:
                    print(f"Found a path after  {k} iterations")
                    G.append(qgoal)
                    print(len(G))
                    edges.append([G.index(qnew), G.index(qgoal)])
                    return G, edges
        print("No path found!")
        return None, None

    def fill_path(self, vertices, edges):
        edges.reverse()
        path = [edges[0][1]]
        next_v = edges[0][0]
        i = 1
        while next_v != 0:
            while edges[i][1] != next_v:
                i += 1
            path.append(edges[i][1])
            next_v = edges[i][0]
            i = 1
        path.append(0)
        edges.reverse()
        path.reverse()
        return vertices, edges, path


    def smoothing(self, C, G, path):
        qfirst_idx = 0
        qlast_idx = len(path) - 1
        qfirst = G[path[qfirst_idx]]
        qlast = G[path[qlast_idx]]
        temp_path = [path[qlast_idx]]

        while qlast_idx > 0:
            dist = qfirst.dist(qlast)
            self.max_depth = round(math.log(dist, 2)) + 1
            while self.is_segment_free_bisection(qfirst, qlast, C, 0) is not True:
                if qfirst_idx + 1 == qlast_idx:
                    break
                qfirst_idx += 1
                qfirst = G[path[qfirst_idx]]
            print(qfirst_idx, qlast_idx)
            temp_path = temp_path + [path[qlast_idx]]
            qlast_idx = qfirst_idx
            qlast = G[path[qlast_idx]]
            qfirst_idx = 0
            qfirst = G[path[qfirst_idx]]
        
        temp_path = temp_path + [path[qfirst_idx]]
        smooth_path = temp_path[::-1] # reverse the path
        return smooth_path
    
    def plot(self, grid_map, states, edges, path):
        plt.figure(figsize=(10, 10))
        plt.matshow(grid_map)

        for i,v in enumerate(states):
            plt.plot(v.y, v.x, "+w")
            # plt.text(v.y, v.x, i, fontsize=14, color="w")

        for e in edges:
            plt.plot(
                [states[e[0]].y, states[e[1]].y],
                [states[e[0]].x, states[e[1]].x],
                "--g",
            )

        for i in range(1, len(path)):
            plt.plot(
                [states[path[i - 1]].y, states[path[i]].y],
                [states[path[i - 1]].x, states[path[i]].x],
                "r",
            )
        # Start
        plt.plot(states[0].y, states[0].x, "r*")
        # Goal
        plt.plot(states[-1].y, states[-1].x, "b*")
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    # Read input from command line
    image_path = str(sys.argv[1]) # file path
    K = int(sys.argv[2]) # number of iterations
    delta_q = int(sys.argv[3]) # step
    p = float(sys.argv[4]) # probability q_rand to be q_goal
    qstart_x = int(sys.argv[5]) # qstart coordinates
    qstart_y = int(sys.argv[6])
    qgoal_x = int(sys.argv[7]) # qgoal coordinates
    qgoal_y = int(sys.argv[8])

    max_depth = round(math.log(delta_q, 2)) + 1 # max iteration for considering a segment free or not
    min_dist = 5

    # Load grid map
    image = Image.open(image_path).convert("L")
    grid_map = np.array(image.getdata()).reshape(image.size[0], image.size[1]) / 255
    # binarize the image
    grid_map[grid_map > 0.5] = 1
    grid_map[grid_map <= 0.5] = 0
    # Invert colors to make 0 -> free and 1 -> occupied
    grid_map = (grid_map * -1) + 1
    # Show grid map
    plt.figure(figsize=(10, 10))
    plt.matshow(grid_map)
    plt.colorbar()
    plt.show()

    path = []
    rrt = RRT(delta_q, p, max_depth, min_dist)
    G, edges = rrt.sample(grid_map, K, p, qstart_x, qstart_y, qgoal_x, qgoal_y)
    G, edges, path = rrt.fill_path(G, edges)
    path = rrt.smoothing(grid_map, G, path)
    
    rrt.plot(grid_map, G, edges, path)
    