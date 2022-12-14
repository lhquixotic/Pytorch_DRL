import os 
import sys
import pickle
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
import gym


from agents.AIf_agents.AIf import AIf
from agents.DQN_agents.DQN import DQN



env_name = "Cart_Pole"
# results_file_name = 'results/data_and_graphs/{}_Results_Data.pkl'.format(env_name)
results_file_name = 'results/data_and_graphs/Cart_Pole_Results_Data.pkl'

print("Loading results file:{} ".format(results_file_name))

f = open(results_file_name,'rb')
results = pickle.load(f)

print(len(results["DQN"]))

config = Config()
config.seed = 1
config.environment = gym.make("CartPole-v1")
config.num_episodes_to_run = 600
config.file_to_save_data_results = "results/data_and_graphs/Cart_Pole_Results_Data.pkl"
config.file_to_save_results_graph = "results/data_and_graphs/Cart_Pole_Results_Graph.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 3
config.use_GPU = True
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False

AGENTS = [AIf,DQN]
results_plotter_trainer = Trainer(config, AGENTS)
results_plotter_trainer.visualise_preexisting_results()