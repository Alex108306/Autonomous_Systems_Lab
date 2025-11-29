import numpy as np
import matplotlib.pyplot as plt

# define model
class MapEnv():
    # define constant inside object MapEnv for each action (0="up", 1="down", 2="right", 3="left")
    ACTION = {"down": 0, "up": 1, "right": 2, "left": 3,}

    def __init__(self, size: int=4, obstacles_percent: float=0.5, stochastic: bool=False) -> None: # default is deterministic!
        self.shape = (size, size)
        self.n_states = size**2
        self.n_actions = len(MapEnv.ACTION)
        self.stochastic = stochastic

        # Create environment map
        self.map = np.zeros((size, size))
        
        # add N obstacles
        N = int(size**2 * obstacles_percent)
        for _ in range(N):
            self.map[np.random.randint(0, size), np.random.randint(0, size)] = 1.0
        
        # Add Goal
        self.goal = (np.random.randint(0, size), np.random.randint(0, size))
        self.goal_id = self.goal[0] * size + self.goal[1]
        self.map[self.goal[0], self.goal[1]] = 0.0 # ensure goal is free!
        
        # Set initial pose to (0, 0)
        self.map[0,0] = 0.0 # ensure init is free!
        self.pose = [0, 0]

    def plot(self) -> None:
        # Plot map
        str_value = []
        for o in self.map.flatten():
            if o == 1.0:
                str_value.append('■')
            else:
                str_value.append(' ')

        plot = np.array(str_value).reshape(self.shape)
        plot[self.goal[0], self.goal[1]] = '★' # set current goal
        plot[self.pose[0], self.pose[1]] = 'O'  # set current pose
        print(plot, '\n')

    def state(self) -> int:
            return self.pose[0] * self.shape[0] + self.pose[1]

    def is_valid(self, state: int) -> bool:
        return self.map.flatten()[state] != 1

    def action(self, a: int) -> int:
        # Move to the next state and compute reward
        motions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        p = self.pose.copy()
        p[0] = (p[0] + motions[a][0]) % self.shape[0]
        p[1] = (p[1] + motions[a][1]) % self.shape[1]
        
        # if the next state is not an obstacle, move to the next state
        if self.map[p[0], p[1]] != 1:
            self.pose = p
        done = False
        if self.pose[0] == self.goal[0] and self.pose[1] == self.goal[1]:
            done = True
        # TODO: return o if done
        reward = -1
        if done:
            reward=0
        return reward, done
    
    def sthocastic_action(self, a: int) -> int:
        # 0.8 chance to do the action, 0.2 chance to do a random action
        if np.random.rand() < 0.8: # --> 0.80% chance to do the action
            return self.action(a)
        else:
            return self.action(np.random.randint(0, 4)) # --> 0.5% to do chosen action + 15% to do a random action 
    
    def model(self, state: int, action: int) -> tuple:
        x = state // self.shape[0]
        y = state % self.shape[1]
        if self.map[x, y] != 1:
            if not self.stochastic:
                self.pose = [x, y] # set initial pose
                r, _ = self.action(action)
                return ([self.state()], [r], [1.0])
            else:
                # stochastic model
                s1s = []
                rs = []
                probs = []
                for i in range(self.n_actions):
                    # print(f"Action: {i} from state: {state} executed action is {action}")
                    self.pose = [x, y] # set initial pose
                    r, _ = self.action(i)
                    s1s.append(self.state())
                    rs.append(r)
                    if i == action:
                        probs.append(0.85)
                    else:
                        probs.append(0.05)
                return (s1s, rs, probs)
        else:
            print("Error: the state is an obstacle!")
            return None

    def step(self, a: int) -> int:
        # Execute deterministic or sthocastic actions
        if not self.stochastic:
            reward, done = self.action(a)
        else:
            reward, done = self.sthocastic_action(a)
        
        # if assumed full observation, return current state, otherwise, return current observation
        return self.state(), reward, done
        # return self.observation(), reward, done
    
    def reset(self) -> None:
        self.pose = [0, 0]
        return self.state()