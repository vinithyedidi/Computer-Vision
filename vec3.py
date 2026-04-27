# utility class to help with 3-dimensional math
from math import sqrt
import numpy as np

class Vec3():
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


    # arithmetic
    def __add__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
        return Vec3(self.x + other, self.y + other, self.z + other)

    def __radd__(self, other):
        return self.__add__(other)


    def __sub__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
        return Vec3(self.x - other, self.y - other, self.z - other)

    def __rsub__(self, other):
        return self.__sub__(other)


    def __mul__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        return Vec3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return self.__mul__(other)


    def __truediv__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x / other.x, self.y / other.y, self.z / other.z)
        return Vec3(self.x / other, self.y / other, self.z / other)


    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    # inplace arithmetic
    def __iadd__(self, other):
        if isinstance(other, Vec3):
            self.x += other.x; self.y += other.y; self.z += other.z
        else:
            self.x += other; self.y += other; self.z += other
        return self


    def __imul__(self, other):
        if isinstance(other, Vec3):
            self.x *= other.x; self.y *= other.y; self.z *= other.z
        else:
            self.x *= other; self.y *= other; self.z *= other
        return self


    # Vector math
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z


    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )


    def norm_sq(self):
        return self.dot(self)


    def norm(self):
        return sqrt(self.norm_sq())


    def normalize(self):
        return self.__truediv__(self.norm())


    def near_zero(self):
        return self.norm_sq() < 1e-16

    @classmethod
    def from_array(cls, arr):
        return cls(arr[0], arr[1], arr[2])


    def to_array(self):
        return np.array([self.x, self.y, self.z])


    def __repr__(self):
        return f'Vec3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})'
