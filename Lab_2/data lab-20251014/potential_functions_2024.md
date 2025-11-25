# Potential Functions

This document will guide you through the practical work related to motion planning algorithms based on potential functions. This lab has three parts: 

* The first one consists on programming an attraction and a repulsive functions, based on the *brushfire* algorithm, and combine them. The second part focuses on the implementation of the gradient descent algorithm used to find the path from any starting position to the goal given a total potential function. The last part consist on implementing the *wave-front* planner to compute the path avoiding the local minima that can appear when combining the attraction and repulsive functions.

All the code has to be programmed in Python.

---

## Grid map environment

We are going to use grayscale images to define our grid map environments. In Python, the `PIL` library can be used to load a map image and the `matplotlib` library can be used to show it. Transform the image to a 2D `numpy` array to simplify its manipulation. Note that when a grayscale image is used as an environment, 0 is black and it is used to represent an obstacle while 1 (or 255) is white and it is normally used to represent the free space. Therefore, we need to binarize the loaded image to ensure that there are no intermediate values as well as to invert the values to have 0 as open space and 1 as obstacles, that is the definition of *free* and *occupied* space that we are going to use. 

**Python code snip**

```python
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Load grid map
image = Image.open('map0.png').convert('L')
grid_map = np.array(image.getdata()).reshape(image.size[0], image.size[1])/255
# binarize the image
grid_map[grid_map > 0.5] = 1
grid_map[grid_map <= 0.5] = 0
# Invert colors to make 0 -> free and 1 -> occupied
grid_map = (grid_map * -1) + 1
# Show grid map
plt.matshow(grid_map)
plt.colorbar()
plt.show()
```

The following maps with the proposed *start* and *goal* loacations are provided:

| grid map name | start     | goal                 |
|---------------|-----------|----------------------|
| map0.png      | (10, 10)  | (110, 40) & (90, 70) |

You may try also

| grid map name | start     | goal      |
|---------------|-----------|-----------|
| map1.png      | (60, 60)  | (90, 60)  | 
| map2.png      | (8, 31)   | (139, 38) |
| map3.png      | (50, 90)  | (375, 375)|

---

## Part 1

### Attraction function: Quadratic Potential Function

An attraction function creates an attraction potential from any point in the `gridMap` to the  `goal`. This potential field can be defined as:

$$
    U_{att}(q) = \frac{1}{2} \zeta d^2(q, q_{goal})
$$

Where:
- $q$: robot configuration.
- $q_{goal}$: goal configuration.
- $d(q, q_{goal})$: Euclidean distance between robot and goal.
- $\zeta$: positive scaling factor to controls the strength of the pull.


Distances can be computed assuming or a 4-Point/8-Point connectivity function.

$$
D_4 = \left[
    \begin{array}{ccc}
    4. & 3. & 2. & 3. & 4. \\
    3. & 2. & 1. & 2. & 3. \\
    2. & 1. & Goal & 1. & 2. \\
    3. & 2. & 1. & 2. & 3. \\   
    4. & 3. & 2. & 3. & 4. 
    \end{array}
\right]
~
D_8 = \left[
    \begin{array}{ccc}
    2. & 2. & 2. & 2. & 2. \\
    2. & 1. & 1. & 1. & 2. \\
    2. & 1. & Goal & 1. & 2. \\
    2. & 1. & 1. & 1. & 2. \\   
    2. & 2. & 2. & 2. & 2. 
    \end{array}
\right]
$$

#### Exercise 1: 
Implement a function that computes the attraction function given a `grid_map` and a `goal` position. The output of this function is a 2D matrix with the same size as the input `grid_map` in which each cell contains the attraction potential to the goal. Generate an image with the attraction function.

> *Example:*

<img src="data/Attraction%20Potential%20Function.png" width="400px"/>

### Repulsive function

The *Repulsive Potential* function is used to keep the robot away from obstacles. The function is defined as:
$$
    U_{rep}(q) =
    \begin{cases}
        \frac{1}{2} \eta \left( \frac{1}{D(q)} - \frac{1}{Q^*} \right)^2, & D(q) \leq Q^* \\
        0,                                                                & D(q) > Q^*
    \end{cases}
$$

Where:
- $D(q)$: distance between the robot and any obstacle.
- $Q^*$: threshold distance beyond which repulsive potential is zero.
- $\eta$: positive constant to control the repulsive force strength.

To compute the repulsive potential, the distance to the nearest obstacle has to be computed. This can be done using the *brushfire* algorithm. The brushfire algorithm is a simple algorithm that starts from the obstacles and propagates a wave-like front until it reaches the free space. The distance to the obstacles is computed as the wave propagates. The algorithm can be implemented using a queue to store the cells that have to be visited. The algorithm pseudocode is shown next:

<pre>
1: <b>brushfire_algorithm</b>(map)
2:   Initialize map d, with obstacle cells in the environment to 1 and free cells to 0
3:   Create a queue L of all boundary cells adjacent to obstacles
4:   <b>while</b> L <b>not empty</b>
5:     Pop the front element t from L
6:     <b>for</b> each neighboring cell n of t
7:       <b>if</b> d(n) = 0
8:         Set d(n) = d(t) + 1
9:         Add n to the queue L
10:      <b>endIf</b>
11:    <b>endFor</b>
12:  <b>endWhile</b>
13:  <b>return</b> the distance map d
</pre>

#### Exercise 2:
Implement a function for the `brushfire` algorithm that given a `grid_map` as input returns a 2D map of distances, with the same size as the input `grid_map`, in which each cell contains the distance to the nearest obstacle. Generate an image with the distance to the nearest obstacle.

> *Example:*

<img src="data/Distance to Nearest Obstacle.png" width="400px"/>

#### Exercise 3:
Implement a function to compute the repulsive function given a map of distances and a `Q` value. The output of this function is a 2D matrix with the same size as the input map of distances in which each cell contains the repulsive potential to the obstacles. Generate an image with the repulsive function


> *Example:*

<img src="data/Repulsive Potential Function.png" width="400px"/>

### Total Potential Function: combining the Attraction and Repulsive functions

The total potential function combines attractive and repulsive potentials. The robot follows the gradient of the total potential function that is defined as:
$$
    U(q) = U_{att}(q) + U_{rep}(q)
$$

#### Exercise 4:
Implement a function that combines the attraction and repulsive functions. The output of this function is a 2D matrix with the same size as the input `grid_map` in which each cell contains the total potential function. Generate an image with the total potential function

> *Example:*

<img src="data/Total potential Function.png" width="400px"/>

## Part 2

### Gradient Descent
Once a potential function is defined, the robot can follow the negative gradient to reach the goal. The gradient of the potential function given a 2D gridmap is defined as: Given the potential value for each cell as a grid, to compute the gradient for a poarticular cell, the difference between the potential value of this cell and its neighbors (using connectivity 8 or connectivity 4) has to be computed.

Example:

$$
    \left[\begin{matrix}
            10 & 12 & 15            & 18 & 20 \\
            8  & 11 & 14            & 16 & 19 \\
            5  & 9  & \color{red}13 & 15 & 17 \\
            3  & 6  & 10            & 12 & 14 \\
            2  & 5  & 7             & 9  & 11 \\
        \end{matrix}\right]
    \quad \rightarrow \text{G.D.} \rightarrow
    \left[\begin{matrix}
            11 - 13 & 14 - 13 & 16 - 13 \\
            9 - 13  & 13 - 13 & 15 - 13 \\
            6 - 13  & 10 - 13 & 12 - 13
        \end{matrix}\right]
    ~=~
    \left[\begin{matrix}
            -2              & 1  & 3  \\
            -4              & 0  & 2  \\
            \color{green}-7 & -3 & -1
        \end{matrix}\right]
$$

A pseudo code to compute the gradient descent is shown next:

<pre>
<b>Gradient Descent (map, q_start)</b>
1: <b>initialize</b> q(0) = q_start
2: <b>while</b> || ∇U(q(i)) || > ε
3:   Compute the gradient: ∇U(q(i))
4:   Update the configuration:
     q(i+1) = q(i) - α ∇U(q(i))
5:   Increment i
6: <b>endWhile</b>
7: <b>return</b> the final configuration q(i)
</pre>


#### Exercise 5:
Implement a gradient descent function to compute the path from $q_{start}$ to the $q_{goal}$ already used to create the *total potential function*. The output of this function is list of traversed cells as well as an image with the path drawn on the grid map.
To test it, use `map0.png` with the start position at $(10, 10)$ and the goal position at $(110, 40)$ first and $(90, 70)$ later. What happens? Why?

> *Example:*

<img src="data/Gradient Descent.png" width="400px"/>

## Part 3

### Wave-front planner
The *Wave-front Planner* is a grid-based path planning algorithm that finds a path from $q_{start}$ to $q_{goal}$ by expanding a wave-like front from the goal, propagating through free space, and labeling cells with distance values. It is similar to the *Brushfire* algorithm, but expanding from the goal instead of form obstacles. It computes a distance field where each cell value represents the minimum number of steps required to reach the goal. Once the grid is labeled, the robot only have to follow the gradient of decreasing values to reach the goal.

The `wave-front` algorithm is detailed in the book *Principles of Robot Motion, Howie Choset et al.*, and a pseudocode is shown next:

<pre>
1: <b>wavefront_planner_connect_4</b>(map, goal)
2:   motions = [left, right, up, down]
3:   value = 2
4:   map[goal] = value
5:   queue = [goal]
6:   <b>while</b> queue <b>not empty</b>
7:     value++
8:     new_queue = []
9:     <b>for</b> p <b>in</b> queue
10:      <b>for</b> m <b>in</b> motions
11:        <b>if</b> isValid(p + m, map)
12:          map[p + m] = value
13:          new_queue.push(p + m)
14:        <b>endIf</b>
15:      <b>endFor</b>
16:    <b>endFor</b>
17:      queue = new_queue
18:  <b>endWhile</b>
19:  <b>return</b> map
</pre>

The `isValid` function checks that the position `p + m` is inside the `map` and the value of `map[p + m]` do not corresponds to an obstacle. 

The result of this algorithm is a new map (a 2D matrix) with the same size than the original grid map in which each cell (or pixel) contains the distance to the goal taking into account the obstacles. 

An example of execution is shown next. In this map 0's represents free space, 1's are obstacles and the 2 indicates the goal position:
<pre>
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 1. 1. 1. 1. 0. 0. 0.]
 [0. 0. 1. 1. 1. 1. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 2. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
</pre>

Running the wavefront planner with connectivity 4, the following attraction function is obtained:

<pre>
 [17. 16. 15. 14. 13. 12. 11. 10. 11. 12.]
 [16. 15. 14. 13. 12. 11. 10.  9. 10. 11.]
 [17. 16.  1.  1.  1.  1.  1.  8.  9. 10.]
 [16. 15.  1.  1.  1.  1.  1.  7.  8.  9.]
 [15. 14.  1.  4.  3.  4.  5.  6.  7.  8.]
 [14. 13.  1.  3.  2.  3.  4.  5.  6.  7.]
 [13. 12.  1.  4.  3.  4.  5.  6.  7.  8.]
 [12. 11.  1.  5.  4.  5.  6.  7.  8.  9.]
 [11. 10.  1.  6.  5.  6.  7.  8.  9. 10.]
 [10.  9.  8.  7.  6.  7.  8.  9. 10. 11.]
</pre>

#### Exercise 6:
Implement a wave-front planner function. Given a `grid_map` as an image and a `goal` position, the output of this algorithm is an `attraction function` that takes into account the obstacles in the environemt. Generate an image with the attraction function.

> *Example:*

<img src="data/Wave-front Map.png" width="400px"/>

### Finding the path

Once the attraction function is implemented, finding a path from any cell to the goal is very simple. Wherever the robot starts, it moves to the neighbouring cell that has the smallest distance to the goal. The process is repeated until the robot reaches the goal. 
You can use the following pseudocode to implement this algorithm:

<pre>
<b>Input:</b> 2D grid with distance values d, start position q, goal position q_goal

 1: <b>While</b> current position q ≠ q_goal:
 2:   Initialize min_neighbor = ∞
 3:   <b>For</b> each neighbor n of q:
 4:     <b>If</b> d(n) < min_neighbor:
 5:       min_neighbor = d(n)
 6:       next_position = n
 7:     <b>EndIf</b>
 8:   <b>EndFor</b>
 9:   Move to next_position
10: <b>EndWhile</b>
</pre>

#### Exercise 7:
Implement a function that given an `find_path` given an `attraction function` (from the wave-front planner algorithm) and a `start` position. The output of this function is a list of positions that the robot has to follow to reach the goal. Generate an image with the path from the start to the goal.

> *Example:*

<img src="data/Wave-Front Path.png" width="400px"/>


## Submission

Submit a report in PDF and a Python Interactive Notebook (only one file!!!) with all the functions implemented. The script must have the name: `potential_functions_YOUR_NAME.ipynb`. 
In the report, explain in detail, and with graphical information, the work done in all sections, show diferent execution examples, discuss about the parameters (e.g., best value of `Q` for a particular environment, diferences between connect-8/4 distance, ...). Explain also the problems you found. You might want to test your algorithm using other environments than the ones provided. **BE SURE** that your functions follows the defined interface! (i.e., inputs and outputs).

## WARNING:

We encourage you to help or ask your classmates for help, but the direct copy of a lab will result in a failure (with a grade of 0) for all the students involved. 

It is possible to use functions or parts of code found on the internet only if they are limited to a few lines and correctly cited (a comment with a link to where the code was taken from must be included). 

**Deliberately copying entire or almost entire works will not only result in the failure of the laboratory but may lead to a failure of the entire course or even to disciplinary action such as temporal or permanent expulsion of the university.** [Rules of the evaluation and grading process for UdG students.](https://tinyurl.com/54jcp2vb)

---

<sup>
Narcís Palomeras
Last review September 2024.
</sup>

