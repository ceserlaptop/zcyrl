import math

import numpy as np


def normalize_angle(angle):
    """Normalize an angle in degree to :math:`[0, 360)`."""
    return (angle + 180) % 360.0 - 180.0


# environment
def angle_diff(angle1, angle2):
    """Calculate the difference between two angles."""
    return abs(normalize_angle(angle1 - angle2))


# coverage_0
def angle_sub(angle1, angle2):
    return ((angle1 - angle2) + 180.0) % 360.0 - 180.0


def eva_cost(R_pos, head_angle, T_pos):
    head_angle = np.radians(head_angle)
    # 计算RR_向量与向量RT之间的夹角
    RR_ = np.array([math.cos(head_angle), math.sin(head_angle)])
    RT = T_pos - R_pos
    d_RT = np.linalg.norm(RT)
    cos_ = np.dot(RR_, RT) / d_RT
    theta = np.arccos(cos_)
    s = (18 * theta / math.pi - 6)
    angle_cost = s / (1 + abs(s)) + 1
    return angle_cost / 4 + d_RT


def polar2cartesian(rho, phi):
    r"""Convert polar coordinates to cartesian coordinates **in degrees**, element-wise.

    .. math::
        \operatorname{polar2cartesian} ( \rho, \phi ) = \left( \rho \cos_{\text{deg}} ( \phi ), \rho \sin_{\text{deg}} ( \phi ) \right)
    """  # pylint: disable=line-too-long

    phi_rad = np.deg2rad(phi)
    return rho * np.array([np.cos(phi_rad), np.sin(phi_rad)])


def arctan2_deg(y, x):
    r"""Element-wise arc tangent of y/x **in degrees**.

    .. math::
        \operatorname{arctan2}_{\text{deg}} ( y, x ) = \frac{180}{\pi} \arctan \left( \frac{y}{x} \right)
    """

    return np.rad2deg(np.arctan2(y, x))


def arcsin_deg(x):
    r"""Trigonometric inverse sine **in degrees**, element-wise.

    .. math::
        \arcsin_{\text{deg}} ( x ) = \frac{180}{\pi} \arcsin ( x )
    """

    return np.rad2deg(np.arcsin(x))


# CoverageWorld
def point_to_line_distance(A, B, C):

    x1, y1 = A
    x2, y2 = B
    x3, y3 = C

    # 计算AB向量
    AB = B - A
    # 计算AC向量
    AC = C - A

    # 计算AC在AB上的投影点D
    projection_length = np.dot(AC, AB) / np.dot(AB, AB)
    D = A + projection_length * AB

    # 计算C到D的距离
    distance = np.linalg.norm(C - D)

    # 判断投影点是否在AB线段上
    in_plag = min(x1, x2) <= x3 <= max(x1, x2) and min(y1, y2) <= y3 <= max(y1, y2)

    return distance, in_plag


class Vector2D:  # pylint: disable=missing-function-docstring
    """2D Vector."""

    def __init__(self, vector=None, norm=None, angle=None, origin=None):
        self.origin = origin
        self._vector = None
        self._angle = None
        self._norm = None
        if vector is not None and norm is None and angle is None:
            self.vector = np.asarray(vector, dtype=np.float64)
        elif vector is None and norm is not None and angle is not None:
            self.angle = angle
            self.norm = norm
        else:
            raise ValueError

    @property
    def vector(self):
        if self._vector is None:
            self._vector = polar2cartesian(self._norm, self._angle)
        return self._vector

    @vector.setter
    def vector(self, value):
        self._vector = np.asarray(value, dtype=np.float64)
        self._norm = None
        self._angle = None

    @property
    def x(self):
        return self.vector[0]

    @property
    def y(self):
        return self.vector[-1]

    @property
    def endpoint(self):
        return self.origin + self.vector

    @endpoint.setter
    def endpoint(self, value):
        endpoint = np.asarray(value, dtype=np.float64)
        self.vector = endpoint - self.origin

    @property
    def angle(self):
        if self._angle is None:
            self._angle = arctan2_deg(self._vector[-1], self._vector[0])
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = normalize_angle(float(value))
        self._vector = None

    @property
    def norm(self):
        if self._norm is None:
            self._norm = np.linalg.norm(self._vector)
        return self._norm

    @norm.setter
    def norm(self, value):
        angle = self.angle
        self._norm = abs(float(value))
        self._vector = None
        if value < 0.0:
            self.angle = angle + 180.0

    def copy(self):
        return Vector2D(vector=self.vector.copy(), origin=self.origin)

    def __eq__(self, other):
        assert isinstance(other, Vector2D)

        return self.angle == other.angle

    def __ne__(self, other):
        return not self == other

    def __imul__(self, other):
        self.norm = self.norm * other

    def __add__(self, other):
        assert isinstance(other, Vector2D)

        return Vector2D(vector=self.vector + other.vector, origin=self.origin)

    def __sub__(self, other):
        assert isinstance(other, Vector2D)

        return Vector2D(vector=self.vector - other.vector, origin=self.origin)

    def __mul__(self, other):
        return Vector2D(norm=self.norm * other, angle=self.angle, origin=self.origin)

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        return Vector2D(norm=self.norm / other, angle=self.angle, origin=self.origin)

    def __pos__(self):
        return self

    def __neg__(self):
        return Vector2D(vector=-self.vector, origin=self.origin)

    def __array__(self):
        return self.vector.copy()
