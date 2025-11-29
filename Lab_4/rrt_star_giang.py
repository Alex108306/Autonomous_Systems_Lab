import numpy as np
import math
from matplotlib import pyplot as plt
from PIL import Image
import argparse

from Point import Point

def load_map(path_to_image: str):
    # Load grid map
    image = Image.open(f"{path_to_image}").convert("L")
    grid_map = np.array(image.getdata()).reshape(image.size[0], image.size[1]) / 255
    # binarize the image
    grid_map[grid_map > 0.5] = 1
    grid_map[grid_map <= 0.5] = 0
    # Invert colors to make 0 -> free and 1 -> occupied
    grid_map = (grid_map * -1) + 1

    return grid_map

def plot(grid_map, states, edges, path, smooth_path, title):
    plt.figure(figsize=(10, 10))
    plt.matshow(grid_map, fignum=0)
    plt.title("Iteration: " + str(title))
    # Start
    plt.plot(states[0].y, states[0].x, "r*", label = "Start")
    # Goal
    plt.plot(states[-1].y, states[-1].x, "g*", label = "Goal")

    for i,v in enumerate(states):
        plt.plot(v.y, v.x, "+w")
        # plt.text(v.y, v.x, i, fontsize=14, color="w")

    for e in edges:
        plt.plot(
            [states[e[0]].y, states[e[1]].y],
            [states[e[0]].x, states[e[1]].x],
            "--g",
        )

    path_x = [states[i].x for i in path]
    path_y = [states[i].y for i in path]
    plt.plot(path_y, path_x, "r", label="Path")
    # for i in range(1, len(path)):
    #     plt.plot(
    #         [states[path[i - 1]].y, states[path[i]].y],
    #         [states[path[i - 1]].x, states[path[i]].x],
    #         "r",
    #     )

    sp_x = [states[i].x for i in smooth_path]
    sp_y = [states[i].y for i in smooth_path]
    plt.plot(sp_y, sp_x, "b", label="Smoothed Path")
    # for i in range(1, len(smooth_path)):
    #     plt.plot(
    #         [states[smooth_path[i - 1]].y, states[smooth_path[i]].y],
    #         [states[smooth_path[i - 1]].x, states[smooth_path[i]].x],
    #         "b",
    #     )
    
    # Plot with legend
    plt.legend(loc ='upper right')
    plt.show()

def fill_path(vertices, edges, edge_goal):
    edges.reverse()
    path = [edge_goal[0][1]]
    next_v = edge_goal[0][0]
    i = 0
    while next_v != 0:
        while edges[i][1] != next_v:
            i += 1
        path.append(edges[i][1])
        next_v = edges[i][0]
        i = 0
    path.append(0)
    edges.reverse()
    path.reverse()
    return vertices, edges, path

def calculate_distance(path, states):
    dist = 0
    for i in range(1, len(path)):
        dist += states[path[i - 1]].dist(states[path[i]])
    
    return dist

def print_path(path, states, mode):
    print(f"{mode} PATH to follow:")
    for i in range(0, len(path)):
        print(states[path[i]])

class RRT_Star:

    def __init__(self, grid_map, K, delta_q, p, max_distance, q_start, q_goal):
        self.c_space = grid_map
        self.K = K
        self.delta_q = delta_q
        self.p = p
        self.max_distance = max_distance
        self.q_start = q_start
        self.q_goal = q_goal

    def run_RRT_Star(self):
        current_iter = 0
        current_dist = 0
        edge_goal = []
        path_first_time = []
        check_first = False
        path_converge = []
        min_dist = 5
        G_vertex = [self.q_start]
        G_edge = []
        J = [0.0]
        threshold = 5
        self.log_delta_q = np.log2(self.delta_q)
        self.log_max_distance = np.log2(self.max_distance)

        c_free_space = []
        for i, j in np.ndindex(self.c_space.shape):
            if self.c_space[i][j] == 0:
                c_free_space.append(Point(i, j))

        for k in range(self.K):
            q_rand = self.RAND_CONF(c_free_space, self.p, self.q_goal)
            id, q_near = self.NEAREST_VERTEX(q_rand=q_rand, G=G_vertex)
            q_new = self.NEW_CONF(q_near=q_near, q_rand=q_rand, delta_q=self.delta_q)
            if q_new not in G_vertex and self.IS_FREE_SEGMENT(q_near=q_near, q_new=q_new, c_space=self.c_space, depth=0, max_depth=self.log_delta_q):
                G_vertex, G_edge, J = self.REWIRE_QNEAR_TO_QNEW(G_vertex, G_edge, id, q_near, q_new, J)
                G_vertex, G_edge, J = self.REWIRE_QNEW_FROM_QNEAR(G_vertex, G_edge, q_new, J)
                if check_first == True and self.DISTANCE(q_new=q_new, q_goal=self.q_goal) < min_dist :
                    _, _, path = fill_path(G_vertex, G_edge ,edge_goal)
                    new_smooth_path = self.smoothing_function(path, G_vertex)
                    new_dist = calculate_distance(new_smooth_path, G_vertex)
                    if new_dist < current_dist:
                        if current_dist - new_dist < threshold:
                            current_iter = k + 1
                            break
                        current_dist = new_dist

                if check_first == False and self.DISTANCE(q_new=q_new, q_goal=self.q_goal) < min_dist:
                        cost_q_goal = J[G_vertex.index(q_new)] + q_new.dist(self.q_goal)
                        J.append(cost_q_goal)
                        G_vertex.append(self.q_goal)
                        edge_goal = [(len(G_vertex) - 2,len(G_vertex) - 1)]
                        G_edge.append(edge_goal[0])
                        path_first_time = [G_vertex.copy(), G_edge.copy(), k + 1]
                        check_first = True
                        _, _, path = fill_path(path_first_time[0], path_first_time[1], edge_goal)
                        smooth_path = self.smoothing_function(path, path_first_time[0])
                        current_dist = calculate_distance(smooth_path, path_first_time[0])


        if current_iter == 0:
            current_iter = self.K
        path_converge = [G_vertex, G_edge, current_iter]
        return path_first_time, path_converge, edge_goal
    
    def REWIRE_QNEAR_TO_QNEW(self, G_vertex, G_edge, idx, q_near, q_new, J):
        id_min = idx
        cost_q_new = J[idx] + q_near.dist(q_new)
        J.append(cost_q_new)
        Q_near = self.NEAR(G_vertex, q_new)
        for id, q in Q_near:
            if self.IS_FREE_SEGMENT(q, q_new, self.c_space, depth=0, max_depth=self.log_max_distance) and (J[id] + q.dist(q_new) < J[-1]):
                id_min = id
                J[-1] = J[id] + q.dist(q_new)
        
        G_vertex.append(q_new)
        G_edge.append((id_min, len(G_vertex) - 1))

        return G_vertex, G_edge, J
    
    def REWIRE_QNEW_FROM_QNEAR(self, G_vertex, G_edge, q_new, J):
        Q_near = self.NEAR(G_vertex, q_new)
        for id, q in Q_near:
            if self.IS_FREE_SEGMENT(q, q_new, self.c_space, depth=0, max_depth=self.log_max_distance) and (J[-1] + q.dist(q_new) < J[id]):
                J[id] = J[-1] + q.dist(q_new)
                id_parent = self.PARENT(G_edge, id)
                G_edge.remove((id_parent, id))
                G_edge.append((len(G_vertex) - 1, id))
        
        return G_vertex, G_edge, J

    def PARENT(self, G_edge, q_child_idx):
        for edge in G_edge:
            if(edge[1] == q_child_idx):
                return edge[0]
    
    def smoothing_function(self, path, G_vertex):
        smooth_path = [path[-1]]
        current_idx = path[-1]
        first = 0
        while current_idx != path[0]:
            while self.IS_FREE_SEGMENT(G_vertex[current_idx], G_vertex[path[first]], self.c_space, depth=0, max_depth=np.log2(G_vertex[current_idx].dist(G_vertex[path[first]]))) == False:
                first += 1
            
            current_idx = path[first]
            first = 0
            smooth_path.append(current_idx)

        smooth_path.reverse()
        return smooth_path

    def NEAR(self, G_vertex, q_new):
        Q_near = []
        for id in range(len(G_vertex)):
            point = G_vertex[id]
            if q_new.dist(point) <= self.max_distance:
                Q_near.append((id, point))
        
        return Q_near

    def RAND_CONF(self, c_free_space, p, q_goal):
        if np.random.rand() < p:
            return q_goal

        else:
            return np.random.choice(c_free_space)

    def NEAREST_VERTEX(self, q_rand, G):
        q_near = G[0]
        nearest_dist = G[0].dist(q_rand)
        idx = 0
        for i in range(len(G)):
            point = G[i]
            dist = q_rand.dist(point)
            if dist < nearest_dist:
                q_near = point
                idx = i
                nearest_dist = dist

        return idx, q_near

    def NEW_CONF(self, q_near, q_rand, delta_q):
        q_near_vector = q_near.vector(q_rand)
        if q_near_vector.norm() < delta_q:
            return q_rand
        q_near_vector_unit = q_near_vector.unit()
        q_new = q_near.__add__(q_near_vector_unit.scale(self.delta_q))
        return Point(round(q_new.x), round(q_new.y))
    
    def IS_FREE_SEGMENT(self, q_near, q_new, c_space, depth, max_depth):
        if depth >= max_depth:
            return True

        if self.POINT_COLLIDED(q_new, c_space):
            return False
        
        q_mid = (q_near.__add__(q_new)).__truediv__(2)

        if self.POINT_COLLIDED(q_mid, c_space):
            return False

        left_segment_free = self.IS_FREE_SEGMENT(q_near, q_mid, c_space, depth + 1, max_depth)
        if left_segment_free == False:
            return False
        right_segment_free = self.IS_FREE_SEGMENT(q_mid, q_new, c_space, depth + 1, max_depth)
        return right_segment_free
    
    def POINT_COLLIDED(self, q, c_space):
        if q.x < 0 or q.x > c_space.shape[0] or q.y < 0 or q.y > c_space.shape[1]:
            return True
        x = round(q.x)
        y = round(q.y)
        if x == c_space.shape[0] - 1 or y == c_space.shape[1] - 1:
            return c_space[x][y] == 1
        
        return c_space[x][y] == 1 or c_space[x+1][y] == 1 or c_space[x+1][y+1] == 1 or c_space[x][y+1] == 1 # round up error compensation

    def DISTANCE(self, q_new, q_goal):
        return q_goal.dist(q_new)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_grid_map', type=str, default='Data/map0.png', help='Path to the grid map image file')
    parser.add_argument('K', type=int, default=10000, help='K parameter')
    parser.add_argument('delta_q', type=int, default=10, help='Delta Q parameter')
    parser.add_argument('p', type=float, default=0.2, help='Probability sample at goal position')
    parser.add_argument('max_distance', type=float, default=30, help='Maximum distance for q_new to consider a near vertex to q_near')
    parser.add_argument('start_x', type=int, default=10, help='Start x position')
    parser.add_argument('start_y', type=int, default=10, help='Start y position')
    parser.add_argument('goal_x', type=int, default=90, help='Goal x position')
    parser.add_argument('goal_y', type=int, default=70, help='Goal y position')
    args = parser.parse_args()

    grid_map = load_map(args.path_to_grid_map)

    start = Point(args.start_x, args.start_y)
    goal = Point(args.goal_x, args.goal_y)

    K = args.K
    delta_q = args.delta_q
    p = args.p
    max_distance = args.max_distance

    rrt = RRT_Star(grid_map=grid_map, K=K, delta_q=delta_q, p=p, max_distance=max_distance, q_start=start, q_goal=goal)
    path_first_found, path_converge, edge_goal = rrt.run_RRT_Star()

    _, _, path_first = fill_path(path_first_found[0], path_first_found[1] ,edge_goal)
    smooth_path_first = rrt.smoothing_function(path_first, path_first_found[0])
    dist_orin_path = calculate_distance(path_first, path_first_found[0])
    dist_smooth_path = calculate_distance(smooth_path_first, path_first_found[0])
    # print(f"Distance original path at {path_first_found[2]} Iterations: ", dist_orin_path)
    print(f"Distance smooth path at {path_first_found[2]} Iterations: ", dist_smooth_path)
    # print_path(path_first, path_first_found[0], f"At {path_first_found[2]} iterations original")
    # print_path(smooth_path_first, path_first_found[0], f"At {path_first_found[2]} iterations smooth")
    plot(grid_map, path_first_found[0], path_first_found[1], path_first, smooth_path_first, path_first_found[2])

    _, _, path_final = fill_path(path_converge[0], path_converge[1], edge_goal)
    smooth_path_final = rrt.smoothing_function(path_final, path_converge[0])
    dist_orin_path = calculate_distance(path_final, path_converge[0])
    dist_smooth_path = calculate_distance(smooth_path_final, path_converge[0])
    # print(f"Distance original path at {path_converge[2]} Iterations: ", dist_orin_path)
    print(f"Distance smooth path at {path_converge[2]} Iterations: ", dist_smooth_path)
    # print_path(path_final, path_converge[0], f"At {path_converge[2]} iterations original")
    # print_path(smooth_path_final, path_converge[0], f"At {path_converge[2]} iterations smooth")
    plot(grid_map, path_converge[0], path_converge[1], path_final, smooth_path_final, path_converge[2])

    # Finetune parameters: 
    # Map 0: k=1000, delta_q=5, p=0.2, max_range=30, start=(10,10), goal=(90,70)
    # Map 1: k=3000, delta_q=10, p=0.2, max_range=20, start=(60,60), goal=(90,60)
    # Map 2: k=20000, delta_q=10, p=0.2, max_range=30, start=(8,31), goal=(139,38)
    # Map 3: k=4000, delta_q=15, p=0.2, max_range=40, start=(50,90), goal=(375,375)
