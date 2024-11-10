# Programed by Z.Zhao
# 考虑连通保持的覆盖场景
import math
import operator

import numpy as np

from experiments.utils import Vector2D, normalize_angle
from multiagent.CoverageWorld import CoverageWorld
from multiagent.core import Agent, Landmark, Obstacle
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self, angle_cover=120, r_cover=0.2, r_comm=0.4, comm_r_scale=0.9, comm_force_scale=5.0):
        self.num_agents = 4
        self.num_pois = 5
        self.num_obstacle = 10
        self.env_range = 1
        self.pos_agents = [[[x, x], [x, -x], [-x, -x], [-x, x]] for x in [0.45 * r_comm]][0]

        self.r_cover = r_cover
        self.r_comm = r_comm
        self.angle_cover = angle_cover
        self.size = 0.02
        self.m_energy = 3.0
        self.max_speed = 0.5
        self.min_speed = 0.1
        self.max_angle_speed = 360

        self.rew_cover = 50.0  # 每单位能量覆盖奖励
        self.rew_done = 500.0
        self.rew_collision = -20.0
        self.rew_out = -100.0

        self.comm_r_scale = comm_r_scale  # r_comm * comm_force_scale = 计算聚合力时的通信半径
        self.comm_force_scale = comm_force_scale  # 连通保持聚合力的倍数

    def make_world(self):
        world = CoverageWorld(self.comm_r_scale, self.comm_force_scale)
        world.collaborative = True

        world.agents = [Agent() for _ in range(self.num_agents)]  # 代表UAV, size为覆盖面积
        world.landmarks = [Landmark() for _ in range(self.num_pois)]  # 代表待覆盖的POI
        world.obstacles = [Obstacle() for _ in range(self.num_obstacle)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent_%d" % i
            agent.collide = False  # 智能体之间不会发生碰撞
            agent.silent = True  # 能够进行通信
            agent.size = self.size
            agent.r_cover = self.r_cover
            agent.r_comm = self.r_comm
            agent.angle_cover = self.angle_cover
            agent.max_speed = self.max_speed
            agent.min_speed = self.min_speed
            agent.max_angle_speed = self.max_angle_speed  # 一个时间片最大偏转36°，dt 0.1
            agent.target_id = -1
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "poi_%d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = self.size
            landmark.m_energy = self.m_energy  # 待覆盖次数
            landmark.label = -1
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = "obstacle_%d" % i
            obstacle.transmittance = 0
            obstacle.collide = False
            obstacle.movable = False
            obstacle.size = self.size

        self.reset_world(world)
        return world

    def reset_world(self, world):
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.05, 0.15, 0.05])
            agent.cover_color = np.array([0.05, 0.25, 0.05])
            agent.comm_color = np.array([0.05, 0.35, 0.05])
            agent.comm_color = np.array([0.05, 0.35, 0.05])
            agent.state.p_pos = np.array(self.pos_agents[i])
            agent.state.p_vel = np.array([self.min_speed, 0])  # 控制率为三维
            agent.state.p_angle = np.random.uniform(-180, 180)
            agent.poi_info = np.zeros((6, 4))

        obstacle_pos = self.generate_obstacle()
        for i, obstacle in enumerate(world.obstacles):
            obstacle.color = np.array([0, 0, 0])
            obstacle.size = obstacle_pos[i][2]
            obstacle.state.p_pos = obstacle_pos[i][0:2]
            obstacle.state.p_vel = np.zeros(world.dim_p)

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.energy = 0.0
            landmark.consume = 0.0
            landmark.done, landmark.just = False, False  # done用于判断单点是否完成覆盖，just用于判断完成覆盖会是否给已经赋予过奖励

        # 随机生成点的位置和角度，避开障碍区域
        generated = 0
        while generated < self.num_pois:
            random_pos = np.random.uniform(-0.8 * self.env_range, 0.8 * self.env_range, (1, 2))

            for i, obstacle in enumerate(world.obstacles):
                dis = np.linalg.norm(obstacle.state.p_pos - random_pos[0])
                # 位于障碍物区域内
                if dis < obstacle.size + self.env_range * 0.01:
                    continue

            world.landmarks[generated].state.p_pos = random_pos[0]
            world.landmarks[generated].state.p_angle = np.random.uniform(-180, 180)
            generated += 1

    def reward(self, agent, world):
        rew = 0.0
        for poi in world.landmarks:
            if poi.just:
                poi.just = False
                rew += self.rew_cover * poi.consume
                poi.consume = 0.0

        # 全部覆盖完成
        if all([poi.done for poi in world.landmarks]):
            rew += self.rew_done

        # 出界惩罚
        for ag in world.agents:
            abs_pos = np.abs(ag.state.p_pos)
            rew += np.sum(abs_pos[abs_pos > 1] - 1) * self.rew_out  # 对出界部分的长度进行计算
            if (abs_pos > 1.2).any():
                rew += self.rew_out

        # 相互碰撞惩罚
        for i, ag in enumerate(world.agents):
            for j, ag2 in enumerate(world.agents):
                if i < j:
                    dist = np.linalg.norm(ag.state.p_pos - ag2.state.p_pos)
                    if dist < 0.2:
                        rew += self.rew_collision

        # 碰撞惩罚
        for ag in world.agents:
            for obstacle in world.obstacles:
                dis = np.linalg.norm(ag.state.p_pos - obstacle.state.p_pos)
                if dis < obstacle.size + ag.size + 0.1:
                    rew += self.rew_collision

        return rew

    # 新观测（策略扩展）
    def observation(self, agent, world):
        info_agents = []
        for other in world.agents:
            if other is agent:
                continue
            info_agents.append(other.state.p_pos - agent.state.p_pos)
            info_agents.append([normalize_angle(other.state.p_angle - agent.state.p_angle)])

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [[agent.state.p_angle]] +
                              info_agents + agent.poi_info.tolist())

    def done(self, agent, world):
        for ag in world.agents:
            abs_pos = np.abs(ag.state.p_pos)
            # p_pos是二维的，abs_pos也是二维的，如果其中超过1.5.则越界
            if (abs_pos > 1.5).any():
                return True
        return all([poi.done for poi in world.landmarks])

    def generate_obstacle(self):
        """产生不重叠的障碍物"""
        while True:
            flag = True
            # 产生随机的位置和大小
            pos = np.random.uniform(-self.env_range * 0.88, self.env_range * 0.88, (self.num_obstacle, 2))
            size = np.random.uniform(self.env_range * 0.02, self.env_range * 0.1, (self.num_obstacle, 1))
            # 验证产生的随机位置和大小是否满足条件:任意两个障碍物不重叠
            for i in range(self.num_obstacle):
                for j in range(i + 1, self.num_obstacle):
                    dis = np.linalg.norm(np.array(pos[i]) - np.array(pos[j]))
                    if dis <= size[i][0] + size[j][0] + self.env_range * 0.1:
                        flag = False
                        break
            if flag:
                obstacle_pos = np.concatenate((np.array(pos), np.array(size)), axis=1)
                return obstacle_pos

# def angle_sub(angle1, angle2):
#     return ((angle1 - angle2) + 2 * math.pi) % (2 * math.pi)


# def eva_cost(R_pos, head_angle, T_pos):
#     # 计算RR_向量与向量RT之间的夹角
#     RR_ = np.array([math.cos(head_angle), math.sin(head_angle)])
#     RT = T_pos - R_pos
#     d_RT = np.linalg.norm(RT)
#     cos_ = np.dot(RR_, RT) / d_RT
#     theta = np.arccos(cos_)
#     s = (18 * theta / math.pi - 6)
#     angle_cost = s / (1 + abs(s)) + 1
#     return angle_cost / 4 + d_RT
