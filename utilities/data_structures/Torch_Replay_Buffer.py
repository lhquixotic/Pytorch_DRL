import random
import torch
import numpy as np

class Torch_Replay_Buffer(object):
    """Replay buffer (in torch datatype) to store past experiences that the agent can then use for training data"""
    
    def __init__(self, buffer_size, batch_size, obs_shape, seed, device=None):

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.observations = torch.empty([buffer_size]+[dim for dim in obs_shape],dtype=torch.float32,device=self.device)
        self.actions = torch.empty(buffer_size,dtype=torch.int64,device=self.device)
        self.rewards = torch.empty(buffer_size,dtype=torch.int8,device=self.device)
        self.dones = torch.empty(buffer_size,dtype=torch.int8,device=self.device)
        self.next_observations = torch.empty([buffer_size]+[dim for dim in obs_shape],dtype=torch.float32,device=self.device)    
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        # self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.push_count = 0

    def position(self):
        return self.push_count % self.buffer_size

    def add_experience(self, obs, actions, rewards, next_obs, dones):
        """Adds experience(s) into the replay buffer"""
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            if isinstance(actions,list): exp_size = len(actions)
            else: exp_size = 1
            self.observations[self.position():self.position()+exp_size] = torch.from_numpy(obs)
            self.actions[self.position():self.position()+exp_size] = actions
            self.rewards[self.position():self.position()+exp_size] = rewards
            self.next_observations[self.position():self.position()+exp_size] = torch.from_numpy(next_obs)
            self.dones[self.position():self.position()+exp_size] = dones
        else:
            if isinstance(actions,list): exp_size = len(actions)
            else: exp_size = 1
            self.observations[self.position():self.position()+exp_size] = torch.from_numpy(obs)
            self.actions[self.position():self.position()+exp_size] = actions
            self.rewards[self.position():self.position()+exp_size] = rewards
            self.next_observations[self.position():self.position()+exp_size] = torch.from_numpy(next_obs)
            self.dones[self.position():self.position()+exp_size] = dones
        self.push_count += exp_size
   
    def sample(self, num_experiences=None):
        """Draws a random sample of experience from the replay buffer"""
        if num_experiences is not None: batch_size = num_experiences
        else: batch_size = self.batch_size
        pick_range = min(self.push_count,self.buffer_size)
        random_index = random.sample(range(pick_range,k=batch_size))
        obs = self.observations[np.array(random_index)]
        actions = self.actions[np.array(random_index)]
        rewards = self.rewards[np.array(random_index)]
        next_obs = self.next_observations[np.array(random_index)]
        dones = self.dones[np.array(random_index)]
        return obs, actions, rewards, next_obs, dones
            
