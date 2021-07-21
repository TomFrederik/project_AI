import gym

class ClusteredActions(gym.Wrapper):
    def __init__(self, env, action_centroids):
        ''' 
        Basically copied from MineRL Research Track BC baseline.
        Wrapper around an action space, that remaps a discrete action to the corresponding centroid.
        The wrapped environment accepts a positive integer action and maps it to the corresponding centroid.
        Args:
            env - gym environment instance
            action_centroids - centroids returned by KMeans on action data
        '''
        super().__init__(env)

        # save centroids
        self.action_centroids = action_centroids
        
        # get num centroids
        self.n_clusters = len(self.action_centroids)
        print(f'Wrapping environment to use {self.n_clusters} discrete actions instead..')

        # save env
        self.base_env = env

        # modify action space to discrete --> choose between centroids
        self.action_space = gym.spaces.Discrete(len(self.action_centroids))

    def step(self, action):
        # remap action to vector
        action_vec = self.remap_action(action)
        
        # take step
        return self.base_env.step(action_vec)
    
    def remap_action(self, action):
        # re-map discrete action to centroid
        action_vec = {'vector':self.action_centroids[action]}
        
        return action_vec