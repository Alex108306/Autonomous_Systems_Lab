import numpy as np

# print Policy as arrows
def print_policy(env, policy):
    policy_symbols = {
        1: '↑',  # up
        2: '➜',  # right
        0: '↓',  # down
        3: '←',  # left
    }
    str_policy = []
    for a, o in zip(policy, env.map.flatten()):
        if o == 1:
            str_policy.append('■')
        else:
            str_policy.append(policy_symbols[a])
    plot = np.array(str_policy).reshape(env.shape)
    plot[env.goal[0], env.goal[1]] = '★'
    print(plot, '\n')

# Print Value function
def print_value_function(env, V):
    str_value = []
    for v, o in zip(V, env.map.flatten()):
        if o == 1:
            str_value.append('  ■ ')
        else:
            str_value.append(f'{v:.1f}')
    plot = np.array(str_value).reshape(env.shape)
    plot[env.goal[0], env.goal[1]] = '  ★ '
    print(plot, '\n')
