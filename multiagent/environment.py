import math
import operator

import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np

from experiments.utils import eva_cost


def angle_sub(angle1, angle2):
    return ((angle1 - angle2) + 2 * math.pi) % (2 * math.pi)


class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True, safe_control=False):

        self.viewer = None
        self.render_geoms_xform = None
        self.render_geoms = None
        self.world = world
        self.agents = self.world.policy_agents
        self.landmarks = world.landmarks
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0
        self.stepnum = 0

        self.safe_control = safe_control

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space

            u_action_space = spaces.Discrete(world.dim_p + 1)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space

            # total action space
            self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))  # 根据coverage_0中的成员函数observation计算观测向量
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            # gym.space.Box是OpenAI gym中的一个空间类，用来表示具有连续值的状态或动作空间，shape表示空间的形状

        # rendering
        self.shared_viewer = shared_viewer
        self.viewers = [None]
        self._reset_render()

    def get_poi_info(self):
        voronoi_cell = {key: [] for key in range(self.n)}
        for poi in self.landmarks:
            if not poi.done:
                costs = [eva_cost(ag.state.p_pos, ag.state.p_angle, poi.state.p_pos) for ag in self.agents]
                min_index, min_cost = min(enumerate(costs), key=operator.itemgetter(1))
                voronoi_cell[min_index].append(poi)
        for i, ag in enumerate(self.agents):
            if len(voronoi_cell[i]):
                # 进行6分区
                div_cell = {key: [] for key in range(6)}
                for poi in voronoi_cell[i]:
                    # 计算RR_与RT的夹角theta，并根据叉乘的大小判断正负，顺时针为负
                    RR_ = np.array([math.cos(np.radians(ag.state.p_angle)),
                                    math.sin(np.radians(ag.state.p_angle))])
                    RT = poi.state.p_pos - ag.state.p_pos
                    cos_ = np.dot(RR_, RT) / (np.linalg.norm(RT))
                    theta = np.arccos(cos_)
                    if np.cross(RR_, RT) < 0:
                        theta = 2 * math.pi - theta
                    theta = math.degrees(theta)
                    # 计算theta 判断分区
                    index = int(((theta + 30) % 360) // 60)
                    div_cell[index].append(poi)
                # 计算每个分区的最小值
                for idx in range(6):
                    nums = len(div_cell[idx])
                    if nums == 0:
                        ag.poi_info[idx] = np.array([0, 0, 0, 0])
                    else:
                        min_cost = 100
                        min_poi = None
                        for poi in div_cell[idx]:
                            cost = eva_cost(ag.state.p_pos, ag.state.p_angle, poi.state.p_pos)
                            if cost < min_cost:
                                min_cost = cost
                                min_poi = poi
                        ag.poi_info[idx] = np.array([nums, max(min_poi.m_energy - min_poi.energy, 0),
                                                     min_poi.state.p_pos[0] - ag.state.p_pos[0],
                                                     min_poi.state.p_pos[1] - ag.state.p_pos[0]])
            else:
                ag.poi_rew = np.zeros((6, 4))

    def step(self, action_n):  # action_n为4*3的列表
        obs_n = []
        self.agents = self.world.policy_agents

        # set action for each agent
        if self.safe_control:  # safe为false,暂时不考虑CBF
            self.update_CBF(action_n)
        else:
            for i, agent in enumerate(self.agents):
                self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()

        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))

        # 周期性更新观测
        if self.stepnum % 15 == 0:
            self.get_poi_info()
        self.stepnum += 1

        # 协作任务共同奖励
        reward_n = [self._get_reward(self.agents[0]) * self.n] * self.n
        done_n = [self._get_done(self.agents[0])] * self.n

        return obs_n, reward_n, done_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        # print("调用environment_get_reward函数")
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        # 根据动力学模型更新控制率
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action

        if agent.movable:
            # 采用做差的方式进行更新，是因为采用Relu网络得到的输出值域在[0,+inf)
            agent.action.u[0] = action[0]  # miu:线性加速度
            agent.action.u[1] = action[1] - action[2]  # omega:角加速度
            sensitivity = 5.0  # 可以将其理解为从输出到控制率的一个映射，参数
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def render1(self, mode='human'):
        if mode not in self.metadata['render.modes']:
            return super().render(mode=mode)

        from multiagent import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(700, 700)
            bound = 1.2
            self.viewer.set_bounds(-bound, bound, -bound, bound)
            margin = rendering.make_polygon(
                np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]), filled=False
            )
            margin.set_linewidth(3)
            self.viewer.add_geom(margin)

        if len(self.viewer.geoms) == 0:
            margin = rendering.make_polygon(
                np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]), filled=False
            )
            margin.set_linewidth(3)
            self.viewer.add_geom(margin)

        for obstacle in self.world.obstacles:
            image = rendering.make_circle(radius=obstacle.size, res=72, filled=True)
            image.add_attr(rendering.Transform(translation=obstacle.state.p_pos))
            image.set_color(*obstacle.color)
            self.viewer.add_geom(image)

        for poi in self.world.landmarks:
            image = rendering.make_circle(radius=poi.size, res=72, filled=True)
            image.add_attr(rendering.Transform(translation=poi.state.p_pos))
            image.set_color(*poi.color)
            self.viewer.add_geom(image)
        # 绘制被遮挡部分
        for agent in self.world.agents:
            polygon = rendering.make_polygon(agent.view_vertices)
            polygon.set_color(0.6, 0.6, 0.0, 0.25)
            self.viewer.add_onetime(polygon)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def render2(self, mode="human"):
        for i in range(len(self.viewers)):
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700, 700)
            else:
                pass

        self.render_geoms = None
        # create rendering geometry创造render的显存
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            # 1. 绘制智能体的位置和目标点的位置
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
            # 2. 绘制智能体的扇形覆盖范围
            for agent in self.world.agents:
                geom_cover = rendering.make_conical(agent.r_cover, np.radians(agent.angle_cover))  # 画扇形
                xform = rendering.Transform()
                geom_cover.set_color(*agent.cover_color, alpha=0.3)
                geom_cover.add_attr(xform)
                self.render_geoms.append(geom_cover)
                self.render_geoms_xform.append(xform)
            # 3. 绘制智能体的扇形遮挡区域
            for agent in self.world.agents:
                agent.update_boundary_vertices(self.world.obstacles)
                geom_view = rendering.make_polygon(agent.view_vertices, filled=True)
                xform = rendering.Transform()
                geom_view.set_color(0.6, 0.6, 0.0, 0.25)
                geom_view.add_attr(xform)
                self.render_geoms.append(geom_view)
                self.render_geoms_xform.append(xform)

            # 4. 将图像加入显示self.viewers中
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
        ##################################################################
        # 新增代码, 显示目标点的覆盖进度
        for geom, entity in zip(self.render_geoms, self.world.entities):
            if 'agent' in entity.name:
                geom.set_color(*entity.color, alpha=0.3)
            else:
                geom.set_color(*entity.color)

        results = []
        for i in range(len(self.viewers)):  # self.viewers为智能体的数目
            from multiagent import rendering
            cam_range = 1.5
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)

            # update geometry positions
            # 1 绘制实体的位置
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # 2. 绘制智能体的扇形覆盖区域
            for e, agent in enumerate(self.agents):
                self.render_geoms_xform[e + len(self.world.entities)].set_translation(agent.state.p_pos[0],
                                                                                      agent.state.p_pos[1])
                self.render_geoms_xform[e + len(self.world.entities)].set_rotation(agent.state.p_angle)

            # 3. 绘制智能体的覆盖遮挡区域
            for e, agent in enumerate(self.agents):
                self.render_geoms_xform[e + len(self.world.entities) + len(self.agents)].set_translation(
                    agent.state.p_pos[0], agent.state.p_pos[1])
                self.render_geoms_xform[e + len(self.world.entities) + len(self.agents)].set_rotation(
                    agent.state.p_angle)

            # # 绘制通信链路
            # for a, ag_a in enumerate(self.agents):
            #     for b, ag_b in enumerate(self.agents):
            #         if b > a:
            #             if np.linalg.norm(ag_a.state.p_pos - ag_b.state.p_pos) < ag_a.r_comm:
            #                 self.viewers[i].draw_line(ag_a.state.p_pos, ag_b.state.p_pos)

            # 框
            self.viewers[i].draw_line([-1, -1], [1, -1])
            self.viewers[i].draw_line([-1, 1], [1, 1])
            self.viewers[i].draw_line([-1, -1], [-1, 1])
            self.viewers[i].draw_line([1, 1], [1, -1])

            # 绘制被遮挡区域
            # for x, agent in enumerate(self.agents):
            #     polygon = rendering.make_polygon(agent.view_vertices, filled=True)
            #     polygon.set_color(0.6, 0.6, 0.0, 0.25)
            #     # polygon.set_color(1.0, 0.0, 0.0, 0.0)
            #     self.viewers[i].add_onetime(polygon)

            # # # 障碍物
            # for obstacle in self.world.obstacle:
            #     self.viewers[i].draw_polygon(obstacle)

            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=(mode == 'rgb_array')))

        return results


    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent:
                        continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700, 700)

        # create rendering geometry创造render的显存
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            # 1. 绘制智能体的位置和目标点的位置
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
            # 2. 绘制智能体的扇形覆盖范围
            for agent in self.world.agents:
                geom_cover = rendering.make_conical(agent.r_cover, np.radians(agent.angle_cover))  # 画扇形
                xform = rendering.Transform()
                geom_cover.set_color(*agent.cover_color, alpha=0.3)
                geom_cover.add_attr(xform)
                self.render_geoms.append(geom_cover)
                self.render_geoms_xform.append(xform)
            # 3. 绘制智能体的扇形遮挡区域
            for agent in self.world.agents:
                agent.update_boundary_vertices(self.world.obstacles)
                geom_view = rendering.make_polygon(agent.view_vertices, filled=True)
                xform = rendering.Transform()
                geom_view.set_color(0.6, 0.6, 0.0, 0.25)
                geom_view.add_attr(xform)
                self.render_geoms.append(geom_view)
                self.render_geoms_xform.append(xform)
            # 新增代码, 绘制uav之间的通信线, 若两个通信则画个线
            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
        ##################################################################
        # 新增代码, 每轮显示agent的r_cover, r_comm, poi的是否完成
        for geom, entity in zip(self.render_geoms, self.world.entities):
            if 'agent' in entity.name:
                geom.set_color(*entity.color, alpha=0.3)
            else:
                geom.set_color(*entity.color)

        results = []
        for i in range(len(self.viewers)):  # self.viewers为智能体的数目
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1.5
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)

            # update geometry positions
            # 1 绘制实体的位置
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # 2. 绘制智能体的扇形覆盖区域
            for e, agent in enumerate(self.agents):
                self.render_geoms_xform[e + len(self.world.entities)].set_translation(agent.state.p_pos[0],
                                                                                      agent.state.p_pos[1])
                self.render_geoms_xform[e + len(self.world.entities)].set_rotation(agent.state.p_angle)

            # 3. 绘制智能体的覆盖遮挡区域
            for e, agent in enumerate(self.agents):
                self.render_geoms_xform[e + len(self.world.entities) + len(self.agents)].set_translation(
                    agent.state.p_pos[0], agent.state.p_pos[1])
                self.render_geoms_xform[e + len(self.world.entities) + len(self.agents)].set_rotation(agent.state.p_angle)

            # # 绘制通信链路
            # for a, ag_a in enumerate(self.agents):
            #     for b, ag_b in enumerate(self.agents):
            #         if b > a:
            #             if np.linalg.norm(ag_a.state.p_pos - ag_b.state.p_pos) < ag_a.r_comm:
            #                 self.viewers[i].draw_line(ag_a.state.p_pos, ag_b.state.p_pos)

            # 框
            self.viewers[i].draw_line([-1, -1], [1, -1])
            self.viewers[i].draw_line([-1, 1], [1, 1])
            self.viewers[i].draw_line([-1, -1], [-1, 1])
            self.viewers[i].draw_line([1, 1], [1, -1])

            # 绘制被遮挡区域
            # for x, agent in enumerate(self.agents):
            #     polygon = rendering.make_polygon(agent.view_vertices, filled=True)
            #     polygon.set_color(0.6, 0.6, 0.0, 0.25)
            #     # polygon.set_color(1.0, 0.0, 0.0, 0.0)
            #     self.viewers[i].add_onetime(polygon)


            # # # 障碍物
            # for obstacle in self.world.obstacle:
            #     self.viewers[i].draw_polygon(obstacle)

            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=(mode == 'rgb_array')))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        # print("调用environment__make_receptor_locations")
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        print("调用environment_BatchMultiAgentEnv类")
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        print("ba_step")
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i + env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        print("bat_reset")
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        print("bat_render")
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n


# def eva_cost(R_pos, head_angle, T_pos):
#     head_angle = np.radians(head_angle)
#     # 计算RR_向量与向量RT之间的夹角
#     RR_ = np.array([math.cos(head_angle), math.sin(head_angle)])
#     RT = T_pos - R_pos
#     d_RT = np.linalg.norm(RT)
#     cos_ = np.dot(RR_, RT) / d_RT
#     theta = np.arccos(cos_)
#     s = (18 * theta / math.pi - 6)
#     angle_cost = s / (1 + abs(s)) + 1
#     return angle_cost / 4 + d_RT
