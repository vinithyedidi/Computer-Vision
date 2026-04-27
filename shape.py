import numpy as np
from abc import ABC, abstractmethod
from texture import Solid_Color
from interval import Interval
from vec3 import Vec3


class Hit_Record:
    def __init__(self):
        self.p = None
        self.normal = Vec3(0, 0, 0)
        self.t = None
        self.u = None
        self.v = None
        self.front_face = None
        self.texture = None

    def set_face_normal(self, r, outward_normal):
        self.front_face = r.direction.dot(outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal


class Primitive(ABC):

    @abstractmethod
    def hit(self, r, ray_t, rec):
        pass


class Quad(Primitive):
    def __init__(self, Q, u, v, texture):
        self.Q = Q
        self.u = u
        self.v = v
        self.texture = texture
        n = u.cross(v)
        nlen = n.norm()
        if nlen < 1e-8:
            raise ValueError('degenerate quad')
        self.normal = n * (1.0 / nlen)
        self.D = self.normal.dot(self.Q)
        self.w = n * (1.0 / n.dot(n))

    def hit(self, r, ray_t, rec):
        denom = self.normal.dot(r.direction)

        if abs(denom) < 1e-8:
            return False

        t = (self.D - self.normal.dot(r.origin)) / denom
        if not ray_t.contains(t):
            return False

        intersection = r.at(t)
        planar_hitpt_vec = intersection - self.Q
        alpha = planar_hitpt_vec.dot(self.u) / self.u.dot(self.u)
        beta  = planar_hitpt_vec.dot(self.v) / self.v.dot(self.v)

        unit_interval = Interval(0, 1)
        if not unit_interval.contains(alpha) or not unit_interval.contains(beta):
            return False

        rec.u = alpha
        rec.v = beta
        rec.t = t
        rec.p = intersection
        rec.texture = self.texture
        rec.set_face_normal(r, self.normal)
        return True


class Sphere(Primitive):
    def __init__(self, center, radius, texture):
        self.center = center
        self.radius = radius
        self.texture = texture

    def hit(self, r, ray_t, rec):
        oc = self.center - r.origin
        a = r.direction.dot(r.direction)
        h = r.direction.dot(oc)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = h * h - a * c

        if discriminant < 0:
            return False

        sqrtd = np.sqrt(discriminant)
        root = (h - sqrtd) / a
        if not ray_t.surrounds(root):
            root = (h + sqrtd) / a
            if not ray_t.surrounds(root):
                return False

        rec.t = root
        rec.p = r.at(rec.t)
        outward_normal = (rec.p - self.center) * (1.0 / self.radius)
        rec.set_face_normal(r, outward_normal)
        rec.u, rec.v = get_sphere_uv(outward_normal)
        rec.texture = self.texture
        return True


def get_sphere_uv(p):
    theta = np.arccos(-p.y)
    phi = np.arctan2(-p.z, p.x) + np.pi
    return phi / (2 * np.pi), theta / np.pi


class Quadric(Primitive):
    def __init__(self, A, b, c, texture):
        self.A = A
        self.b = b
        self.c = c
        self.texture = texture

    def hit(self, r, ray_t, rec):
        # A and b are numpy matrices/arrays, so convert Vec3 to array for @ ops
        o = r.origin.from_array()
        d = r.direction.from_array()
        a = d @ self.A @ d
        b = 2 * (o @ self.A @ d) + self.b @ d
        c = o @ self.A @ o + self.b @ o + self.c

        if np.abs(a) < 1e-8:
            if np.abs(b) < 1e-8:
                print('quadric is a single point')
                return False
            print('quadric is linear')
            root = -c / b
            if not ray_t.surrounds(root):
                return False
        else:
            discriminant = b * b - 4 * a * c
            if discriminant < 0:
                return False
            sqrtd = np.sqrt(discriminant)
            root = (-b - sqrtd) / (2 * a)
            if not ray_t.contains(root):
                root = (-b + sqrtd) / (2 * a)
                if not ray_t.contains(root):
                    return False

        rec.t = root
        rec.p = r.at(root)
        # A and b are numpy — convert rec.p for matrix math, then wrap result
        outward_normal = Vec3.from_array(2 * self.A @ rec.p.to_array() + self.b)
        outward_normal = outward_normal * (1.0 / outward_normal.length())
        rec.set_face_normal(r, outward_normal)
        rec.texture = self.texture
        rec.u = None
        rec.v = None
        return True