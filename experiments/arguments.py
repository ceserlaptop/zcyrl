# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch

import time
import torch
import argparse

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
time_now = time.strftime('%y%m_%d%H%M')


def parse_args():
    parser = argparse.ArgumentParser("reinforcement learning experiments for multi-agent environments")

    # environment
    parser.add_argument("--scenario_name", type=str, default="coverage_0", help="name of the scenario script")
    parser.add_argument("--start_time", type=str, default=time_now, help="the time when start the game")
    parser.add_argument("--max_step_len", type=int, default=120, help="maximum episode lengt4h")
    parser.add_argument("--max_episode", type=int, default=1000, help="maximum episode length")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")

    parser.add_argument("--transmittance", type=float, default=0.0, help="obstacle transmittance,"
                                                                         "在这里设置是对所有障碍物统一进行设置")
    parser.add_argument("--agent_r_cover", type=float, default=0.5, help="覆盖半径")
    parser.add_argument("--obstacle_size_small", type=float, default=0.02, help="obstacle size of small")
    parser.add_argument("--obstacle_size_large", type=float, default=0.1, help="obstacle size of large")
    parser.add_argument("--agent_num", type=int, default=4, help="number of agent")
    parser.add_argument("--obstacle_num", type=int, default=10, help="number of obstacle")
    parser.add_argument("--poi_num", type=int, default=5, help="number of pois")

    # core training parameters
    parser.add_argument("--device", default=device, help="torch device ")
    parser.add_argument("--safe_control", type=bool, default=True, help="adopt the CBF ")
    parser.add_argument("--learning_start_step", type=int, default=10000, help="learning start steps")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max gradient norm for clip")
    parser.add_argument("--learning_fre", type=int, default=100, help="learning frequency")
    parser.add_argument("--tau", type=int, default=0.01, help="how depth we exchange the par of the nn")
    parser.add_argument("--lr_a", type=float, default=0.002, help="learning rate for adam optimizer")
    parser.add_argument("--lr_c", type=float, default=0.002, help="learning rate for adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--memory_size", type=int, default=1e6, help="number of data bufferstored in the memory")
    parser.add_argument("--num_units_1", type=int, default=128, help="number of units in the mlp")
    parser.add_argument("--num_units_2", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--num_units_openai", type=int, default=128, help="number of units in the mlp")

    # checkpointing;
    parser.add_argument("--fre_save_model", type=int, default=500,
                        help="the number of the episode for saving the model")
    parser.add_argument("--save_dir", type=str, default="models", help="directory in which training state and model \
    should be saved")
    parser.add_argument("--old_model_name", type=str,
                        default="coverage_0/coverage_0_05_15_09_00/policy/coverage_0_0515_090614_1000/",
                        help="directory in which training state and model are loaded")
    # parser.add_argument("--old_model_name", type=str, default="models/coverage_0_1220_151236_100000/", help="directory in \
    # which training state and model are loaded")

    # evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=10000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", \
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", \
                        help="directory where plot data is saved")
    return parser.parse_args()
