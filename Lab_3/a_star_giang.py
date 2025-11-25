import numpy as np
import heapq as hq
from matplotlib import pyplot as plt
from PIL import Image
import argparse

class A_star:

    def __init__(self, grid_map, start, goal, h, connectivity):
        self.grid_map = grid_map
        self.start = start
        self.goal = goal
        self.h = h
        self.connectivity = connectivity
    
    def run_A_star_algorithm(self):

        if self.connectivity == 4:
            move_list = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        elif self.connectivity == 8:
            move_list = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]
        
        camefrom = {f"{self.start}": None}

        g_score = np.zeros(shape=self.grid_map.shape)

        for i, j in np.ndindex(g_score.shape):
            g_score[i][j] = np.inf

        g_score[self.start[0]][self.start[1]] = 0

        f_score = np.zeros(shape=self.grid_map.shape)

        for i, j in np.ndindex(f_score.shape):
            f_score[i][j] = np.inf

        f_score[self.start[0]][self.start[1]] = (
            g_score[self.start[0]][self.start[1]] + self.h[self.start[0]][self.start[1]]
        )

        open_set = [(f_score[self.start[0]][self.start[1]], self.start)]
        close_set = set()

        while len(open_set) != 0:
            hq.heapify(open_set)

            current = hq.heappop(open_set)
            close_set.add((current[1][0], current[1][1]))

            if current[1] == self.goal:
                total_cost = current[0]
                return self.reconstruct_path(camefrom, current[1]), total_cost

            for move in move_list:
                
                next_step = [current[1][0] + move[0], current[1][1] + move[1]]

                if (
                    next_step[0] >= self.grid_map.shape[0] 
                    or next_step[0] < 0
                    or next_step[1] >= self.grid_map.shape[1]
                    or next_step[1] < 0
                    or self.grid_map[next_step[0]][next_step[1]] == 1
                ):
                    continue

                tentative_g_score = g_score[current[1][0]][current[1][1]] + self.Euclidean_distance([current[1][0], current[1][1]],[next_step[0], next_step[1]])

                if ((next_step[0], next_step[1]) not in close_set and ((f_score[next_step[0]][next_step[1]], next_step) not in open_set or tentative_g_score < g_score[next_step[0]][next_step[1]])):

                    camefrom[f"{next_step}"] = current[1]

                    g_score[next_step[0]][next_step[1]] = tentative_g_score

                    if (f_score[next_step[0]][next_step[1]], next_step) not in open_set:

                        hq.heappush(
                            open_set,
                            (
                                g_score[next_step[0]][next_step[1]] + self.h[next_step[0]][next_step[1]],
                                next_step,
                            ),
                        )

                    else:
                        open_set[
                            open_set.index((
                                f_score[next_step[0]][next_step[1]],
                                next_step,
                            ))
                        ] = (
                            g_score[next_step[0]][next_step[1]]
                            + self.h[next_step[0]][next_step[1]],
                            next_step,
                        )

                    f_score[next_step[0]][next_step[1]] = g_score[next_step[0]][next_step[1]] + self.h[next_step[0]][next_step[1]]

        return False
    
    def Euclidean_distance(self, a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
    
    def reconstruct_path(self, camefrom, current):
        total_path = [current]
        while camefrom[f"{current}"] is not None:
            current = camefrom[f"{current}"]
            total_path.insert(0, current)
        return total_path
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_grid_map', type=str, default='Data/map0.png', help='Path to the grid map image file')
    parser.add_argument('start_x', type=int, default=10, help='Start x coordinate')
    parser.add_argument('start_y', type=int, default=10, help='Start y coordinate')
    parser.add_argument('goal_x', type=int, default=90, help='Goal x coordinate')
    parser.add_argument('goal_y', type=int, default=70, help='Goal y coordinate')
    args = parser.parse_args()

    start = [args.start_x, args.start_y]
    goal = [args.goal_x, args.goal_y]
    connectivity_list = [4, 8]  # 4 or 8

    # Load grid map
    image = Image.open(args.path_to_grid_map).convert("L")
    grid_map = np.array(image.getdata()).reshape(image.size[0], image.size[1]) / 255
    # binarize the image
    grid_map[grid_map > 0.5] = 1
    grid_map[grid_map <= 0.5] = 0
    # Invert colors to make 0 -> free and 1 -> occupied
    grid_map = (grid_map * -1) + 1
    # Show grid map
    plt.matshow(grid_map)
    plt.plot(start[1], start[0], 'go', markersize = 10, label = "Start")
    plt.plot(goal[1], goal[0], 'r*', markersize = 10, label = "Goal")
    plt.colorbar()
    plt.legend()
    plt.show()

    # Heuristic function h (Euclidean distance)
    h = np.zeros(shape=grid_map.shape)

    for i, j in np.ndindex(h.shape):
        h[i][j] = np.sqrt(np.square(i - goal[0]) + np.square(j - goal[1]))
    
    # Run A* for different connectivity
    for connectivity in connectivity_list:
        print(f"Using connectivity {connectivity}:")

        # A* algorithm
        a_star = A_star(grid_map.copy(), start, goal, h, connectivity)
        path, total_cost = a_star.run_A_star_algorithm()
    
        traj_x = []
        traj_y = []
        for point in path:
            traj_x.append(point[0])
            traj_y.append(point[1])
    
        if path:
            print("Path cost:", total_cost)
            print("Path:", path)
    
            # Visualize path
            plt.matshow(grid_map)
            plt.plot(start[1], start[0], 'go', markersize = 10, label = "Start")
            plt.plot(goal[1], goal[0], 'r*', markersize = 10, label = "Goal")
            plt.plot(traj_y, traj_x, 'r-')
            plt.legend()
            plt.colorbar()
            plt.show()
        else:
            print("No path found.")