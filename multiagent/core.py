# 1.增加连通强制力,
# 1) 对没有和其他agent建立连接的agent, 会受到与他最近的agent之间的拉力
# 2) 若所有agent均有邻居但未达到全连接, 则对当前所有距离里比通信距离大的最小距离添加拉力

import numpy as np
from scipy.interpolate import interp1d

from experiments.utils import normalize_angle, Vector2D, arcsin_deg, polar2cartesian


# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None
        # 位置角度
        self.p_angle = None


# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None


# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.max_angle_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        self.last_angle = None

    @property
    def mass(self):
        return self.initial_mass

    def __sub__(self, other):
        if not isinstance(other, Entity):
            raise NotImplementedError
        return Vector2D(vector=self.state.p_pos - other.state.p_pos, origin=other.state.p_pos)


# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()


class Obstacle(Entity):
    def __init__(self):
        super(Obstacle, self).__init__()

    def obstruct(self, ray, keep_tangential=False, outer=False):
        relative = Vector2D(vector=self.state.p_pos - ray.origin)
        norm = ray.norm
        # 如果障碍物中心到智能体的距离为0，或者距离小于障碍物半径，则返回相反的射线
        if norm == 0.0 or relative.norm < self.size:
            return -ray
        # 如果障碍物中心到智能体的距离大于障碍物半径，则返回原射线
        if relative.norm >= norm + self.size:
            return ray

        inner = np.inner(relative.vector, ray.vector)  # 两个向量做积。即当前向量和智能体与障碍物的连线向量的积

        if inner >= 0.0:
            # 计算两个向量的余弦
            cos = min(1.0, inner / (relative.norm * norm))
            # 计算垂直距离
            perpendicular = relative.norm * np.sqrt(1.0 - np.square(cos))
            # 表示当前向量在障碍物的里面
            if self.size > perpendicular:
                # half_chord则表示向量在圆内的长度（半和弦）
                half_chord = np.sqrt(np.square(self.size) - np.square(perpendicular))
                if not outer:  # 如果不包括障碍物，则新的长度为减去半和弦
                    new_norm = max(0.0, relative.norm * cos - half_chord)
                else:  # 如果包括障碍物，则新的长度为加上半和弦
                    new_norm = max(0.0, relative.norm * cos + half_chord)
                if new_norm < norm:  # 裁剪了向量的长度，则返回新的长度
                    old_ray = ray.vector
                    ray.norm = new_norm
                    if keep_tangential:  # 保持切线
                        radius = ray.endpoint - self.state.p_pos
                        ray.vector = old_ray + radius * (
                                (norm - new_norm) * half_chord / np.square(self.size)
                        )
        return ray


# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        """  智能体的继承属性  """
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        """  智能体的自定义属性  """
        # 要执行的脚本行为
        self.action_callback = None
        self.view_vertices = []
        self.r_cover = 0
        self.angle_cover = 0

        self.boundary = []
        self.sight_range_func = None
        self.sight_range_outer_func = None
        self.obstacles = set()
        self.last_angle = None

    def sight_range_at(self, angle, outer=False):
        angle = normalize_angle(angle)
        if outer:
            return self.sight_range_outer_func(angle)
        return self.sight_range_func(angle)

    def boundary_between(self, outer=False):
        # 计算出当前覆盖的左右边界角度
        # print("角度:", self.state.p_angle, "\n")
        angle_left = normalize_angle(self.state.p_angle - self.angle_cover / 2.0)
        # angle_left = normalize_angle(self.state.p_angle)
        angle_right = angle_left + self.angle_cover
        # print("左角度:", angle_left, "右角度:", angle_right, "正角度:", self.state.p_angle, "覆盖角度:", self.angle_cover, "\n")

        if outer:  # 默认false
            phis_all = self.sight_range_outer_func.x
            rhos_all = self.sight_range_outer_func.y
        else:
            phis_all = self.sight_range_func.x
            rhos_all = self.sight_range_func.y

        # 通过下面的操作，得到在智能体视野范围内的向量角度（phis）和长度（rhos）
        # if angle_right <= +180.0:
        if angle_right <= +180.0 and angle_left >= -180.0:
            # 输出是否同时满足两个命题的布尔值，得到在angle_left和angle_right之间的布尔值排列
            mask = np.logical_and(angle_left < phis_all, phis_all < angle_right)
            phis = phis_all[mask]
            rhos = rhos_all[mask]
        else:
            # if angle_right <= +180.0:
            #     mask1 = np.logical_and(angle_left < phis_all, phis_all <= +180.0)
            #     mask2 = np.logical_and(phis_all > -180.0, phis_all < angle_right)
            #     phis = np.concatenate([phis_all[mask1], phis_all[mask2]])
            #     rhos = np.concatenate([rhos_all[mask1], rhos_all[mask2]])
            # elif angle_left >= -180.0:
            #     mask1 = np.logical_and(angle_left < phis_all, phis_all <= 0.0)
            #     mask2 = np.logical_and(phis_all > -180.0, phis_all < angle_right - 360.0)
            #     phis = np.concatenate([phis_all[mask1], phis_all[mask2]])
            #     rhos = np.concatenate([rhos_all[mask1], rhos_all[mask2]])

            mask1 = np.logical_and(angle_left < phis_all, phis_all <= +180.0)
            mask2 = np.logical_and(phis_all > -180.0, phis_all < angle_right - 360.0)
            phis = np.concatenate([phis_all[mask1], phis_all[mask2]])
            rhos = np.concatenate([rhos_all[mask1], rhos_all[mask2]])
        phis = np.concatenate([[angle_left], phis, [angle_right]])
        rhos = np.concatenate(
            [[self.sight_range_at(angle_left)], rhos, [self.sight_range_at(angle_right)]]
        )
        # print("角度", phis)
        return phis.astype(np.float64), rhos.astype(np.float64)

    def add_obstacle(self, env_obstacles):
        # 过滤障碍物，保留在智能体覆盖域中的障碍物
        obstacles = set(
            filter(
                lambda x: np.linalg.norm(self.state.p_pos - x.state.p_pos) < self.r_cover + x.size,
                filter(lambda obstacle: obstacle is not self, env_obstacles),
            )
        )
        # self.obstacles.update(obstacles)
        # boundary = self.boundary

        boundary = [
            # 向量的长度（距离），角度（弧度制），原点，角度从-180到+180，每隔一度一个向量
            Vector2D(norm=self.r_cover, angle=angle, origin=self.state.p_pos)
            for angle in np.linspace(-180.0, +180.0, num=360, endpoint=False)
        ]
        # 如果该障碍物透过率为1（透明），则不用进行计算
        for obstacle in obstacles:
            if obstacle.transmittance == 1.0:
                continue

            # 得到相对距离
            relative = obstacle - self
            # 如果障碍物尺寸大于相对距离，则智能体在障碍物里面，那么边界就是圆形
            if obstacle.size > relative.norm:
                boundary = [
                    Vector2D(norm=0, angle=angle, origin=self.state.p_pos)
                    for angle in range(-180, 180, 90)
                ]
                break
            # 根据距离和障碍物的大小，计算遮盖角度的一半
            half_opening_angle = arcsin_deg(obstacle.size / relative.norm)
            # 选择覆盖范围和障碍物最远端其中一个作为max_rho
            max_rho = min(self.r_cover, relative.norm + obstacle.size)
            # 覆盖角度范围（左和右）
            angle_left = relative.angle - half_opening_angle
            angle_right = relative.angle + half_opening_angle
            # 把值添加进去
            boundary.extend(
                [
                    Vector2D(
                        norm=self.r_cover, angle=angle_left - 0.01, origin=self.state.p_pos
                    ),
                    Vector2D(
                        norm=self.r_cover, angle=angle_left + 0.01, origin=self.state.p_pos
                    ),
                    Vector2D(
                        norm=self.r_cover, angle=angle_right - 0.01, origin=self.state.p_pos
                    ),
                    Vector2D(
                        norm=self.r_cover, angle=angle_right + 0.01, origin=self.state.p_pos
                    ),
                ]
                + [
                    Vector2D(norm=max_rho, angle=angle, origin=self.state.p_pos)
                    for angle in np.linspace(
                        angle_left,
                        angle_right,
                        num=max(16, int(2 * half_opening_angle)) + 1,
                        endpoint=True,
                    )
                ]
            )

            self.boundary = [obstacle.obstruct(b) for b in boundary]

        # for obstacle in obstacles:
        #     if obstacle.transmittance == 1.0:
        #         continue
        #
        #     boundary = [obstacle.obstruct(b) for b in boundary]

        def interpolate(boundary):
            # 根据角度进行排序
            boundary.sort(key=lambda ray: ray.angle)
            boundary, unfiltered = [], boundary
            # 遍历未过滤的边界
            for ray in unfiltered:
                if len(boundary) > 0 and boundary[-1].angle == ray.angle:  # 表示上一个向量和当前的向量角度一样（即角度发生了重复）
                    if boundary[-1].norm > ray.norm:  # 选一个更短的向量
                        boundary[-1] = ray
                else:
                    boundary.append(ray)

            rhos = [ray.norm for ray in boundary]
            phis = [ray.angle for ray in boundary]
            rhos.append(rhos[0])
            phis.append(phis[0] + 360)

            rhos = np.asarray(rhos, dtype=np.float64)
            phis = np.asarray(phis, dtype=np.float64)

            return interp1d(phis, rhos), boundary

        self.sight_range_func, _ = interpolate(boundary)

    def clear_obstacles(self):
        self.obstacles.clear()

    def update_boundary_vertices(self, env_obstacle):
        self.clear_obstacles()
        self.add_obstacle(env_obstacle)
        phis, rhos = self.boundary_between()
        # 去除在圆心覆盖域外的边界点
        rhos = rhos.clip(min=self.size, max=self.r_cover)
        # if self.name == "agent_0":
        #     # print("角度数据", phis, "距离数据", rhos, "\n")
        #     print("当前角度:", self.state.p_angle, "\n")
        # phi_rad = np.deg2rad(phis)
        # vertices = rhos * np.array([np.cos(phi_rad), np.sin(phi_rad)])
        # print(vertices)
        vertices = polar2cartesian(rhos, phis).transpose()
        #  表示所有的边界点
        # print(vertices)
        vertices = self.state.p_pos + np.concatenate([[[0.0, 0.0]], vertices, [[0.0, 0.0]]])
        # vertices = np.concatenate([[[0.0, 0.0]], vertices, [[0.0, 0.0]]])
        self.view_vertices = vertices


# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.obstacles = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks + self.obstacles

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if (b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if (f_a is not None):
                    if (p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if (f_b is not None):
                    if (p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(
                                                                          entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

            # get collision forces for any contact between two entities

    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):  # 两个都是collide时计算碰撞
            return [None, None]  # not a collider
        if (entity_a is entity_b):
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos  # 切线方向
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k  #
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
