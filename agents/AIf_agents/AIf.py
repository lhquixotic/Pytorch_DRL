from collections import Counter
from distutils.command.config import config
from email import policy
from tracemalloc import start
from turtle import forward

import torch
import random
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F
from nn_builder.pytorch.NN import NN
import numpy as np 
from agents.Base_Agent import Base_Agent
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from utilities.data_structures.Torch_Replay_Buffer import Torch_Replay_Buffer


'''
config.hyperparameters = {
    "AIf_Agents": {
        "batch_size": 256,
        "buffer_size": 40000,
        "tra_lr": 0.001,
        "pol_lr": 0.001,
        "val_lr": 0.001,
        "gamma" : 1.00,
        "update_every_n_steps": 5,

        "epsilon": 0.98,
        "epsilon_decay_rate_denominator": 1,
        "discount_rate": 1,
        "tau": 0.01,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.1,
        "incremental_td_error": 1e-8,
        
        "linear_hidden_units": [30, 15],
        "final_layer_activation": "None",
        "batch_norm": False,
        "gradient_clipping_norm": 0.7,
        "learning_iterations": 1,
        "clip_rewards": False
    }
}
'''

class AIf(Base_Agent):
    """ An active inference agent """
    agent_name = "AIf"
    def __init__(self, config):
        Base_Agent.__init__(self,config)
        
        # Initialize the env params
        self.observation_shape = self.environment.observation_space.shape
        self.observation_size = int(np.prod(self.observation_shape))
        self.state_size = self.observation_size
        self.batch_size = self.hyperparameters["batch_size"]

        # Initialize the replay memory
        # self.memory = Replay_Buffer(self.hyperparameters["buffer_size"],self.hyperparameters["batch_size"],config.seed, self.device)
        self.memory = Torch_Replay_Buffer(self.hyperparameters["buffer_size"],self.hyperparameters["batch_size"],self.observation_shape,self.device)
        

        # Initialize the networks
        self.transition_network = self.create_forward_NN(self.observation_size+1, self.observation_size, [64])
        self.transition_optimizer = optim.Adam(self.transition_network.parameters(),
                                                lr = self.hyperparameters["tra_lr"])
        policy_params = {"final_layer_activation":"SOFTMAX"}       
        self.policy_network = self.create_forward_NN(self.observation_size,self.action_size,[64],hyperparameters=policy_params)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(),
                                                lr = self.hyperparameters["pol_lr"])
        self.value_network = self.create_forward_NN(self.observation_size, self.action_size,[64])
        self.value_optimizer = optim.Adam(self.value_network.parameters(),
                                                lr = self.hyperparameters["val_lr"])

        self.target_network = self.create_forward_NN(self.observation_size, self.action_size,[64])
        self.target_network.load_state_dict(self.value_network.state_dict())

        self.logger.info("Transition network {}.".format(self.transition_network))
        self.logger.info("Policy network {}.".format(self.policy_network))
        self.logger.info("Value network {}.".format(self.value_network))

        self.gamma = self.hyperparameters["gamma"]
        self.beta = self.hyperparameters["beta"]

        # Sample from the replay memory
        self.obs_indices = [2,1,0]
        self.action_indices = [2,1]
        self.reward_indices = [1]
        self.done_indices = [0]
        self.max_n_indices = max(max(self.obs_indices, self.action_indices, self.reward_indices, self.done_indices))+1

    def reset_game(self):
        super(AIf,self).reset_game()
        self.update_learning_rate(self.hyperparameters["tra_lr"],self.transition_optimizer)
        self.update_learning_rate(self.hyperparameters["pol_lr"],self.policy_optimizer)
        self.update_learning_rate(self.hyperparameters["val_lr"],self.value_optimizer)
    

    def step(self):
        while not self.done:
            if self.config.use_GPU:
                self.action = int(self.pick_action().flatten().cpu().numpy())
            else:
                self.action = int(self.pick_action().flatten().numpy())
            self.conduct_action(self.action)
            self.learn()
            self.save_experience()
            self.state = self.next_state
            self.global_step_number += 1
        self.episode_number += 1
            
    def pick_action(self,state=None):
        if state is None: state = self.state
        if isinstance(state,np.int64) or isinstance(state, int): state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2: state = state.unsqueeze(0)
        # action selection
        # self.policy_network.eval()
        with torch.no_grad():
            policy = self.policy_network(state)
        # self.policy_network.train()
        action = torch.multinomial(policy,1)
        self.logger.info("Action chosen {} -- policy network score {}.".format(action,policy))
        return action
        
    def learn(self, experiences=None):
        """ Run a learning iteration for the network """
        
        # Memory check
        # If memory data is not enough
        if self.global_step_number  < self.hyperparameters["batch_size"] + 2*self.max_n_indices:
            return

        # Update the target_network periodly
        if self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0:
            self.target_network.load_state_dict(self.value_network.state_dict())
        
        # Retrieve transition data in mimi batches:
        (obs_batch_t0, obs_batch_t1, obs_batch_t2, action_batch_t0,
         action_batch_t1, reward_batch_t1, done_batch_t2,
         pred_error_batch_t0t1) = self.get_mini_batches()

        # Compute the value network loss:
        value_network_loss = self.compute_value_net_loss(obs_batch_t1, obs_batch_t2,
                                                        action_batch_t1,reward_batch_t1,
                                                        done_batch_t2, pred_error_batch_t0t1)
        
        # Compute the variational free energy:
        VFE = self.compute_VFE(obs_batch_t1,pred_error_batch_t0t1)

        # Reset the gradients:
        self.transition_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        # Compute the gradients:
        VFE.backward()
        value_network_loss.backward()
        self.logger.info("Value network loss -- {}, VFE -- {}".format(value_network_loss.item(),VFE.item()))
        self.log_gradient_and_weight_information(self.value_network,self.value_optimizer)
        # Perform gradient descent:
        self.transition_optimizer.step()
        self.policy_optimizer.step()
        self.value_optimizer.step()

    def compute_value_net_loss(self, obs_batch_t1, obs_batch_t2,
                               action_batch_t1, reward_batch_t1,
                               done_batch_t2, pred_error_batch_t0t1):
        
        with torch.no_grad():
            # Determine the action distribution for time t2:
            policy_batch_t2 = self.policy_network(obs_batch_t2)
            
            # Determine the target EFEs for time t2:
            target_EFEs_batch_t2 = self.target_network(obs_batch_t2)
            
            # Weigh the target EFEs according to the action distribution:
            weighted_targets = ((1-done_batch_t2) * policy_batch_t2 *
                                target_EFEs_batch_t2).sum(-1).unsqueeze(1)
                
            # Determine the batch of bootstrapped estimates of the EFEs:
            EFE_estimate_batch = -reward_batch_t1 + pred_error_batch_t0t1 + self.beta * weighted_targets
        
        # Determine the EFE at time t1 according to the value network:
        EFE_batch_t1 = self.value_network(obs_batch_t1).gather(1, action_batch_t1)
        # print("Debug shape: {},{},{}".format(policy_batch_t2.shape,target_EFEs_batch_t2.shape,weighted_targets.shape))
        # print("Debug shape: {}".format(done_batch_t2.shape))
        # print("EFE shape:{} vs {}".format(EFE_batch_t1.shape,EFE_estimate_batch.shape))

        # Determine the MSE loss between the EFE estimates and the value network output:
        value_net_loss = F.mse_loss(EFE_estimate_batch, EFE_batch_t1)
        
        return value_net_loss

    def compute_VFE(self, obs_batch_t1, pred_error_batch_t0t1):
        
        # Determine the action distribution for time t1:
        policy_batch_t1 = self.policy_network(obs_batch_t1)
        
        # Determine the EFEs for time t1:
        EFEs_batch_t1 = self.value_network(obs_batch_t1).detach()

        # Take a gamma-weighted Boltzmann distribution over the EFEs:
        boltzmann_EFEs_batch_t1 = torch.softmax(-self.gamma * EFEs_batch_t1, dim=1).clamp(min=1e-9, max=1-1e-9)
        
        # Weigh them according to the action distribution:
        energy_batch = -(policy_batch_t1 * torch.log(boltzmann_EFEs_batch_t1)).sum(-1).view(self.memory.batch_size, 1)
        
        # Determine the entropy of the action distribution
        entropy_batch = -(policy_batch_t1 * torch.log(policy_batch_t1)).sum(-1).view(self.memory.batch_size, 1)
        
        # Determine the VFE, then take the mean over all batch samples:
        VFE_batch = pred_error_batch_t0t1 + (energy_batch - entropy_batch)
        VFE = torch.mean(VFE_batch)
        
        return VFE

    def get_mini_batches(self):
        # Retrieve transition data in mini batches
        all_obs_batch, all_actions_batch, reward_batch_t1, done_batch_t2 = self.sample_experiences()
        # print("all obs batch: {}".format(all_obs_batch))
        # Retrieve a batch of observations for 3 consecutive points in time
        obs_batch_t0 = all_obs_batch[:, 0].view([self.batch_size] + [dim for dim in self.observation_shape])
        obs_batch_t1 = all_obs_batch[:, 1].view([self.batch_size] + [dim for dim in self.observation_shape])
        obs_batch_t2 = all_obs_batch[:, 2].view([self.batch_size] + [dim for dim in self.observation_shape])
        
        # Retrieve the agent's action history for time t0 and time t1
        action_batch_t0 = all_actions_batch[:, 0].unsqueeze(1)
        action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1)

        # print("pred_batch:{}, obs_batch:{}".format(pred_batch_t0t1.shape,obs_batch_t1.shape))
        
        # print("\nobs shape : {}, action shape: {}".format(obs_batch_t0.shape,action_batch_t0.shape))
        # print("\ndones shape : {}, action shape: {}".format(done_batch_t2,action_batch_t0.shape))

        # At time t0 predict the state at time t1:
        X = torch.cat((obs_batch_t0, action_batch_t0.float()), dim=1)
        # print("X shape={}".format(X.shape))
        pred_batch_t0t1 = self.transition_network(X)
        # print("pred_batch:{}, obs_batch:{}".format(pred_batch_t0t1.shape,obs_batch_t1.shape))
        # Determine the prediction error wrt time t0-t1:
        pred_error_batch_t0t1 = torch.mean(F.mse_loss(
                pred_batch_t0t1, obs_batch_t1, reduction='none'), dim=1).unsqueeze(1)
        
        return (obs_batch_t0, obs_batch_t1, obs_batch_t2, action_batch_t0,
                action_batch_t1, reward_batch_t1, done_batch_t2, pred_error_batch_t0t1)
    
    def sample_experiences(self):
        
        # Pick indices at random
        end_indices = np.random.choice(min(self.global_step_number, self.hyperparameters["buffer_size"])-self.max_n_indices*2, self.memory.batch_size, replace=False) + self.max_n_indices

        # Correct for sampling near the position where data was last pushed
        for i in range(len(end_indices)):
            if end_indices[i] in range(self.memory.position(), self.memory.position()+ self.max_n_indices):
                end_indices[i] += self.max_n_indices

        ''' Use deque Replay Buffer 
        # memory_list = list(self.memory.memory)
        # print("memory list:{}".format(memory_list))
        # print("index:{},{}".format(start_index,end_index))
        # experiences = memory_list

        # observations = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).to(self.device)
        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).to(self.device)
        # actions = actions.squeeze(-1)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).to(self.device)
        # rewards = rewards.squeeze(-1)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).to(self.device)
        # dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).to(self.device)
        # dones = dones.squeeze(-1)

        # observations,actions,rewards,next_observations, dones = self.memory.sample(self.memory.batch_size+self.max_n_indices*2)
        # print("shape of all obs batch: {}".format(observations.shape))
        # print("shape:{},{},{},{}".format(observations.shape,actions.shape,rewards.shape,dones.shape))

        # end_indices = [ i for i in range(len(observations)-self.memory.batch_size,len(observations))]
        # end_indices = np.array(end_indices)

        # Retrieve the specified indices that come before the end_indices
        '''

        """ Use torch Replay Buffer """



        obs_batch = self.memory.observations[np.array([index-self.obs_indices for index in end_indices])]
        action_batch = self.memory.actions[np.array([index-self.action_indices for index in end_indices])]
        reward_batch = self.memory.rewards[np.array([index-self.reward_indices for index in end_indices])]
        done_batch = self.memory.dones[np.array([index-self.done_indices for index in end_indices])]

        # print("\nllobs shape : {}, action shape: {}".format(obs_batch.shape,action_batch.shape))


        # print("\nshape:{},{},{},{}".format(obs_batch.shape,action_batch.shape,reward_batch.shape,done_batch.shape))

        # Correct for sampling over multiple episodes
        for i in range(len(end_indices)):
            index = end_indices[i]
            for j in range(1, self.max_n_indices):
                if self.memory.dones[index-j]:
                    for k in range(len(self.obs_indices)):
                        if self.obs_indices[k] >= j:
                            obs_batch[i, k] = torch.zeros_like(self.memory.observations[0]) 
                    for k in range(len(self.action_indices)):
                        if self.action_indices[k] >= j:
                            action_batch[i, k] = torch.zeros_like(self.memory.actions[0]) # Assigning action '0' might not be the best solution, perhaps as assigning at random, or adding an action for this specific case would be better
                    for k in range(len(self.reward_indices)):
                        if self.reward_indices[k] >= j:
                            reward_batch[i, k] = torch.zeros_like(self.memory.rewards[0]) # Reward of 0 will probably not make sense for every environment
                    for k in range(len(self.done_indices)):
                        if self.done_indices[k] >= j:
                            done_batch[i, k] = torch.zeros_like(self.memory.dones[0]) 
                    break
                
        return obs_batch, action_batch, reward_batch, done_batch

    def memory_position(self):
        return self.global_step_number % self.hyperparameters["buffer_size"]

    def create_forward_NN(self,input_dim,output_dim,layers_info,hyperparameters=None,override_seed=None):
        default_hyperparameters = {"output_activation": None, "hidden_activations": "relu", 
                                          "final_layer_activation":None,"dropout": 0.0,
                                          "initialiser": "default", "batch_norm": False,
                                          "columns_of_data_to_be_embedded": [],
                                          "embedding_dimensions": [], "y_range": ()}
        if isinstance(input_dim,list): input_dim=input_dim[0]  
        if hyperparameters is None: hyperparameters = default_hyperparameters
        if override_seed: seed=override_seed
        else: seed = self.config.seed

        for key in default_hyperparameters:
            if key not in hyperparameters.keys():
                hyperparameters[key] = default_hyperparameters[key]

        return NN(input_dim=input_dim, layers_info=layers_info+[output_dim],
                  output_activation=hyperparameters["final_layer_activation"],
                  batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
                  hidden_activations=hyperparameters["hidden_activations"], initialiser=hyperparameters["initialiser"],
                  columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
                  embedding_dimensions=hyperparameters["embedding_dimensions"], y_range=hyperparameters["y_range"],
                  random_seed=seed).to(self.device)



