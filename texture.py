from abc import ABC, abstractmethod
import math
from vec3 import Vec3


class Texture(ABC):

    @abstractmethod
    def value(self, u, v, p):
        pass


class Solid_Color(Texture):
    def __init__(self, albedo: Vec3):
        self.albedo = albedo

    def value(self, u, v, p):
        return self.albedo


class Checker_Texture(Texture):
    def __init__(self, scale, even: Vec3, odd: Vec3):
        self.scale = scale
        self.even = Solid_Color(even)
        self.odd = Solid_Color(odd)

    def value(self, u, v, p):
        s = int(math.floor(self.scale * u))
        t = int(math.floor(self.scale * v))
        is_even = (s + t) % 2 == 0
        return self.even.value(u, v, p) if is_even else self.odd.value(u, v, p)