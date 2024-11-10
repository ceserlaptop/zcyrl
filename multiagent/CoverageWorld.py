import math

import numpy as np

from experiments.poltVerticesTest import showVertices
from experiments.utils import angle_diff, Vector2D, arcsin_deg, normalize_angle
from multiagent.core import World
from scipy.interpolate import interp1d


class CoverageWorld(World):
    def __init__(self, comm_r_scale=0.9, comm_force_scale=0.0):
        super(CoverageWorld, self).__init__()
        self.coverage_rate = 0.0  # 每次step后重新计算
        self.connect = False  # 当前是否强连通
        self.dist_mat = np.zeros([4, 4])  # agents之间的距离矩阵, 对角线为1e5
        self.adj_mat = np.zeros([4, 4])  # 标准r_comm下的邻接矩阵, 对角线为0

        # 对连通保持聚合力的修正
        self.contact_force *= comm_force_scale  # 修正拉力倍数
        self.comm_r_scale = comm_r_scale  # 产生拉力的半径 = r_comm * comm_r_scale

        # 在comm_r_scale修正下的强连通和邻接矩阵
        self.connect_ = False  # 用于施加规则拉力的强连通指示
        self.adj_mat_ = np.zeros([4, 4])  # 用于施加规则拉力的邻接矩阵
        self.dt = 0.1
        self.delta = 0.01
        self.sensitivity = 2.0

    def step(self):
        self.update_collision()  # 计算是否与出界，是否发生碰撞
        p_force = [None for _ in range(len(self.agents))]
        p_force = self.apply_action_force(p_force)  # 得到位置控制率
        self.integrate_state(p_force)
        self.update_energy()

    def update_collision(self):
        self.collision = False
        self.outRange = False
        # 检查出界
        for ag in self.agents:
            abs_pos = np.abs(ag.state.p_pos)
            if (abs_pos > 1.2).any():  # 任意一维大于1.2则认为出界
                self.outRange = True
                break

        # 检查无人机碰撞
        for i, ag in enumerate(self.agents):
            if self.collision:
                break
            for j, ag2 in enumerate(self.agents):
                if i < j:
                    dist = np.linalg.norm(ag.state.p_pos - ag2.state.p_pos)
                    if dist < 0.2:
                        self.collision = True
                        break

    def apply_action_force(self, p_force):
        for i, agent in enumerate(self.agents):
            p_force[i] = agent.action.u
        return p_force

    def integrate_state(self, p_force):
        for i, entity in enumerate(self.agents):
            if not entity.movable: continue
            if p_force[i] is not None:
                entity.state.p_vel = p_force[i]
            # 对线速度进行限幅
            if entity.max_speed is not None and np.abs(entity.state.p_vel[0]) > entity.max_speed:
                entity.state.p_vel[0] = entity.max_speed
            if entity.min_speed is not None and np.abs(entity.state.p_vel[0]) < entity.min_speed:
                entity.state.p_vel[0] = entity.min_speed
            # 对角速度进行限幅
            if entity.max_angle_speed is not None and np.abs(entity.state.p_vel[1]) > entity.max_angle_speed:
                entity.state.p_vel[1] = entity.state.p_vel[1] / np.abs(entity.state.p_vel[1]) * entity.max_angle_speed
            # print("角速度限幅", entity.max_angle_speed)
            # 位姿更新
            entity.state.p_angle += entity.state.p_vel[1] * self.dt
            entity.state.p_angle = normalize_angle(entity.state.p_angle)  # 角度变换，角度变成弧度，将角度限制在0-2*pi
            entity.state.p_pos[0] += entity.state.p_vel[0] * math.cos(entity.state.p_angle) * self.dt
            entity.state.p_pos[1] += entity.state.p_vel[0] * math.sin(entity.state.p_angle) * self.dt


    def update_energy(self):
        num_done = 0
        for poi in self.landmarks:
            if poi.done:
                num_done += 1
            else:
                for agent in self.agents:
                    dist = np.linalg.norm(poi.state.p_pos - agent.state.p_pos)
                    # beta为目标位置相对于智能体的方位角，delta为目标方位角和朝向角度之差
                    beta = np.arctan2(poi.state.p_pos[1] - agent.state.p_pos[1],
                                      poi.state.p_pos[0] - agent.state.p_pos[0])
                    beta = normalize_angle(np.degrees(beta))
                    delta = angle_diff(beta, agent.state.p_angle)
                    # 如果小于覆盖半径且朝向角和目标方位角之差小于覆盖角度
                    if dist <= agent.r_cover and delta < agent.angle_cover / 2:
                        # 基于扇形感知的能量函数
                        c1 = max(0, agent.r_cover - dist)
                        c2 = max(0, (agent.angle_cover / 2 - delta)*math.pi/180.0)
                        c3 = max(0, (agent.angle_cover / 2 + delta)*math.pi/180.0)
                        poi.consume += 20 * (c1 * c2 * c3) / (c1 * c2 + c1 * c3 + c2 * c3)

                if poi.consume > 0.0:
                    # 限制覆蓋能量大小，最大為m_energy
                    poi.energy += poi.consume
                    if poi.energy > poi.m_energy:
                        poi.consume -= poi.energy - poi.m_energy
                        poi.energy = poi.m_energy
                    poi.just = True

                if poi.energy >= poi.m_energy:
                    poi.done = True
                    poi.just = True
                    num_done += 1
                    poi.color = np.array([0.25, 1.0, 0.25])
                poi.color = np.array([0.25, 0.25 + poi.energy / poi.m_energy * 0.75, 0.25])
        self.coverage_rate = num_done / len(self.landmarks)

# 计算两个角度之差, 给定的角度在0~2*pi之间
# def angle_diff(angle1, angle2):
#     return abs(angle1 - angle2) if abs(angle1 - angle2) <= 2 * math.pi - abs(angle1-angle2) else (
#             2 * math.pi - abs(angle1-angle2))
