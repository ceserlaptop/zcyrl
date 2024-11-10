import math
import os
import sys

import imageio
import numpy as np

sys.path.append(os.path.abspath("../multiagent-particle-envs/"))
import torch
import torch.nn.functional as F
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
import time

from model import actor_agent, critic_agent
from arguments import parse_args

scenarios_name = "coverage_0"

fileOrder = -2
fire_name = os.listdir("./" + scenarios_name)[fileOrder]
load_path = ("./" + scenarios_name + "/" + fire_name + "/policy/"
             + os.listdir("./" + scenarios_name + "/" + fire_name + "/policy/")[-1]) + "/"

is_save_gif = True
save_gif_path = "./" + scenarios_name + "/gif/coverage_done_line.gif"
save_np_path = "./" + scenarios_name + "/gif/coverage_done_line.csv"
save_poi_path = "./" + scenarios_name + "/gif/poi_pos.csv"

def make_env(scenario_name, arglist):
    """
    create the environment from script
    """
    scenario = scenarios.load(scenario_name + ".py").Scenario(angle_cover=120, r_cover=0.25, r_comm=1,
                                                              comm_r_scale=0.9, comm_force_scale=5.0)
    world = scenario.make_world()
    if arglist.benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            done_callback=scenario.done)
        # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, arglist):
    """ load the model """
    actors_tar = [torch.load(load_path + 'a_c_{}.pt'.format(agent_idx), map_location=arglist.device)
                  for agent_idx in range(env.n)]

    return actors_tar

def enjoy(arglist):
    """
    This func is used for testing the model
    """

    episode_step = 0
    """ init the env """
    env = make_env(arglist.scenario_name, arglist)

    """ init the agents """
    actors_tar = get_trainers(env, arglist)

    """ interact with the env """
    obs_n = env.reset()
    while (1):

        # update the episode step number
        episode_step += 1

        # get action
        action_n = []
        for actor, obs in zip(actors_tar, obs_n):
            model_out, _ = actor(torch.from_numpy(obs).to(arglist.device, torch.float), model_original_out=True)
            action_n.append(F.softmax(model_out, dim=-1).detach().cpu().numpy())

        # interact with env
        obs_n, rew_n, done_n = env.step(action_n)
        # print(episode_step, '\t', env.world.coverage_rate)

        # update the flag
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)

        # reset the env
        if done or terminal:
            print(episode_step, '\t', env.world.coverage_rate)
            break
        # else:
        #     obs_n = env.reset()
        #     episode_step = 0
    return episode_step
        # render the env

        # env.render()
        # time.sleep(0.1)

def save_gif(arglist):
    """
    This func is used for testing the model
    """

    """ init the env """
    env = make_env(arglist.scenario_name, arglist)

    """ init the agents """
    actors_tar = get_trainers(env, arglist)

    """ interact with the env """
    t_start = time.time()
    obs_n = env.reset()
    episode_step = 0
    frames = []
    record = []
    path_data = []
    landmark = []
    for poi in env.landmarks:
        landmark.append(poi.state.p_pos)
    landmark = np.array(landmark)
    np.savetxt(save_poi_path, landmark, delimiter=',')
    while (True):
        if is_save_gif:
            frames.append(env.render(mode='rgb_array')[0])
        else:
            time.sleep(0.2)
            env.render()

        # get action
        action_n = []
        for actor, obs in zip(actors_tar, obs_n):
            model_out, _ = actor(torch.from_numpy(obs).to(arglist.device, torch.float), model_original_out=True)
            action_n.append(F.softmax(model_out, dim=-1).detach().cpu().numpy())

        # interact with env
        obs_n, rew_n, done_n = env.step(action_n)

        if is_save_gif:
            tmp = []
            for ag in env.agents:
                tmp.append(np.array([ag.state.p_pos[0], ag.state.p_pos[1], ag.state.p_angle]))
            pos = np.array(tmp).reshape(-1)
            path_data.append(pos)

        if is_save_gif:
            p = np.array([ag.state.p_pos[:] for ag in env.agents]).reshape(-1)
            v = np.array([ag.state.p_vel[:] for ag in env.agents]).reshape(-1)
            total = np.append(p,v)
            a = np.array([ag.action.u for ag in env.agents]).reshape(-1)
            total = np.append(total, a)
            record.append(total)

        episode_step += 1

        # update the flag
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)

        # 保存gif图片
        if done or terminal:
            if is_save_gif:
                if episode_step >= 400:
                    frames = []
                    record = []
                    path_data = []
                    obs_n = env.reset()
                    episode_step = 0
                    continue

                env.close()
                frames = [frames[0]] * 5 + frames
                frames += [frames[-1]] * 5
                imageio.mimsave(save_gif_path, frames, 'GIF', duration=0.1)
                record = np.array(record)

                # np.savetxt(save_np_path, record, delimiter=',')
                path_data = np.array(path_data)
                np.savetxt(save_np_path, path_data, delimiter=',')
                break
            else:
                obs_n = env.reset()
                episode_step = 0
    print(episode_step)
    print(time.time() - t_start)

if __name__ == '__main__':
    arglist = parse_args()
    # enjoy(arglist)
    save_gif(arglist)
    # steps = []
    # for i in range(100):
    #     num = enjoy(arglist)
    #     steps.append(num)
    # print("-----------------")
    # print(steps)
    # print(np.mean(steps))
