from collections import Counter
from distutils.command.config import config
from email import policy
from tracemalloc import start
from turtle import forward

import torch
import random
from torch.distributions import Normal
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F
from nn_builder.pytorch.NN import NN
import numpy as np 
from agents.Base_Agent import Base_Agent
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from utilities.data_structures.Torch_Replay_Buffer import Torch_Replay_Buffer
from agents.AIf_agents.AIf import AIf

TRAINING_EPISODES_PER_EVAL_EPISODE = 10

class AIf_Continuous(AIf):
    """ An active inference agent """
    agent_name = "AIf"
    def __init__(self, config):
        Base_Agent.__init__(self,config)

        assert self.action_types == "CONTINUOUS", "Action types should not be discrete."

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

        actor_params = {"final_layer_activation":"TANH"}       
        self.actor_network = self.create_forward_NN(self.observation_size,self.action_size*2,[64],hyperparameters=actor_params)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(),
                                                lr = self.hyperparameters["pol_lr"])

        self.value_network = self.create_forward_NN(self.observation_size, self.action_size,[64])
        self.value_optimizer = optim.Adam(self.value_network.parameters(),
                                                lr = self.hyperparameters["val_lr"])

        self.target_network = self.create_forward_NN(self.observation_size, self.action_size,[64])
        self.target_network.load_state_dict(self.value_network.state_dict())

        self.logger.info("Transition network {}.".format(self.transition_network))
        self.logger.info("Actor network {}.".format(self.actor_network))
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
        self.update_learning_rate(self.hyperparameters["pol_lr"],self.actor_optimizer)
        self.update_learning_rate(self.hyperparameters["val_lr"],self.value_optimizer)
    

    def step(self):
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0
        
        while not self.done:
            self.action = self.pick_action(eval_ep)
            self.conduct_action(self.action)
            self.learn()
            self.save_experience()
            self.state = self.next_state
            self.global_step_number += 1
        self.episode_number += 1
            
    def pick_action(self, eval_ep, state=None):
        if state is None: state = self.state
        if isinstance(state,np.int64) or isinstance(state, int): state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2: state = state.unsqueeze(0)

        if eval_ep: action = self.actor_pick_action(state=state,eval=True)
        elif self.global_step_number < self.hyperparameters["min_steps_before_learning"]:
            action = self.environment.action_space.sample()
            print("Picking randonm action: ",action)
        else: action = self.actor_pick_action(state=state)
        if self.add_extra_noise:
            action += self.noise.sample()
        return action
    
    def actor_pick_action(self, state=None, eval=False):
        """Uses actor to pick an action in one of two ways: 1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled 2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling"""
        if state is None: state = self.state
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1: state = state.unsqueeze(0)
        if eval == False: action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]
        
    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        actor_output = self.actor_network(state)
        mean, log_std = actor_output[:, :self.action_size], actor_output[:, self.action_size:]
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  #rsample means it is sampled using reparameterisation trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


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
        self.actor_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        # Compute the gradients:
        VFE.backward()
        value_network_loss.backward()
        self.logger.info("Value network loss -- {}, VFE -- {}".format(value_network_loss.item(),VFE.item()))
        self.log_gradient_and_weight_information(self.value_network,self.value_optimizer)
        # Perform gradient descent:
        self.transition_optimizer.step()
        self.actor_optimizer.step()
        self.value_optimizer.step()

    def compute_value_net_loss(self, obs_batch_t1, obs_batch_t2,
                               action_batch_t1, reward_batch_t1,
                               done_batch_t2, pred_error_batch_t0t1):
        
        with torch.no_grad():
            # Determine the action distribution for time t2:
            policy_batch_t2,_,_  = self.produce_action_and_action_info(obs_batch_t2)
            
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
        policy_batch_t1,_,_ = self.produce_action_and_action_info(obs_batch_t1)
        
        # Determine the EFEs for time t1:
        EFEs_batch_t1 = self.value_network(obs_batch_t1).detach()

        # Take a gamma-weighted Boltzmann distribution over the EFEs:
        boltzmann_EFEs_batch_t1 = torch.softmax(-self.gamma * EFEs_batch_t1, dim=1).clamp(min=1e-9, max=1-1e-9)
        
        # Weigh them according to the action distribution:
        energy_batch = -(policy_batch_t1 * torch.log(boltzmann_EFEs_batch_t1 + 1e-6)).sum(-1).view(self.memory.batch_size, 1)
        
        # Determine the entropy of the action distribution
        entropy_batch = -(policy_batch_t1 * torch.log(policy_batch_t1 + 1e-6)).sum(-1).view(self.memory.batch_size, 1)
        
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

        """ Use torch Replay Buffer """
        obs_batch = self.memory.observations[np.array([index-self.obs_indices for index in end_indices])]
        action_batch = self.memory.actions[np.array([index-self.action_indices for index in end_indices])]
        reward_batch = self.memory.rewards[np.array([index-self.reward_indices for index in end_indices])]
        done_batch = self.memory.dones[np.array([index-self.done_indices for index in end_indices])]

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



