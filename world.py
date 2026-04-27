from shape import Sphere, Hit_Record
from interval import Interval


# Basically just a list of objects in the world with hit() able to call hit detection for each object
class World:
    def __init__(self):
        self.objects = []

    def clear(self):
        self.objects = []

    def add(self, obj):
        self.objects.append(obj)

    def hit(self, r, ray_t, rec):
        hit_anything = False
        closest_so_far = ray_t.max

        for obj in self.objects:
            temp_rec = Hit_Record()
            if obj.hit(r, Interval(ray_t.min, closest_so_far), temp_rec):
                hit_anything = True
                closest_so_far = temp_rec.t
                rec.p = temp_rec.p
                rec.normal = temp_rec.normal
                rec.t = temp_rec.t
                rec.front_face = temp_rec.front_face
                rec.texture = temp_rec.texture
                rec.u = temp_rec.u
                rec.v = temp_rec.v
        return hit_anything