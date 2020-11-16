import utils
import mdp
import numpy as np

def lava_ambiguous_corridor():
    num_rows = 10
    num_cols = 15
    num_states = num_rows * num_cols
   
    state_features = np.array([(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(0,1,1),(1,0,1),(1,1,1)])

    weights = np.array([-100, -100])#np.array([-0.26750391, -0.96355677])#np.array([-.18, -.82])

    weights = weights / np.linalg.norm(weights)
    print(weights)
    gamma = 0.99
    init_dist = np.ones(num_states)
    init_states = [100,4]
    for si in init_states:
        init_dist[si] = 1.0 / len(num_states) * np.ones(num_states)
    term_states = [200]
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env


def negative_sideeffects_goal(num_rows, num_cols, num_features, unseen_feature=False):
    #no terminal random rewards and features

    num_states = num_rows * num_cols

    if unseen_feature:
        assert(num_features >=3)

    #create one-hot features
    f_vecs = np.eye(num_features)
    features = [tuple(f) for f in f_vecs]

    state_features = []
    for i in range(num_states):
        #select all but last two states randomly (last state is goal, second to last state is possibly unseen)
        if unseen_feature:
            r_idx = np.random.randint(num_features-1)
        else:
            r_idx = np.random.randint(num_features-2)
        state_features.append(features[r_idx])
    
    #select goal
    goal_state = np.random.randint(num_states)
    state_features[goal_state] = features[-1]

    
    state_features = np.array(state_features)


    #sample from L2 ball
    weights = -np.random.rand(num_features)
    #set goal as positive
    weights[-1] = +2
    #set unseen as negative
    weights[-2] = -2
    weights = weights / np.linalg.norm(weights)
    
    print("weights", weights)
    gamma = 0.99
    #let's look at all starting states for now
    init_states = [100]
    for si in init_states:
        init_dist[si] = 1.0 / np.ones(num_states) / len(num_states)
   

    #no terminal
    term_states = [goal_state]
    
    mdp_env = mdp.FeaturizedGridMDP(num_rows, num_cols, state_features, weights, gamma, init_dist, term_states)
    return mdp_env
