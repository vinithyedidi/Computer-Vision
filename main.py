import numpy as np

from shape import Sphere, Quad, Quadric
from world import World
from camera import Camera
from texture import Solid_Color, Checker_Texture
from vec3 import Vec3
from numba_renderer import render
from demo import demo


# def generate_N_views(N):
#     world = World()
#
#     # =========================================================
#     # 1. CORNELL-STYLE BOX (keeps geometry stable reference)
#     # =========================================================
#
#     left_wall = Quad(
#         Vec3(-2, -2, -6),
#         Vec3(0, 0, 4),
#         Vec3(0, 4, 0),
#         Solid_Color(Vec3(0.5, 0.5, 0.5))
#     )
#
#     back_wall = Quad(
#         Vec3(-2, -2, -6),
#         Vec3(4, 0, 0),
#         Vec3(0, 4, 0),
#         Solid_Color(Vec3(0.6, 0.6, 0.6))
#     )
#
#     right_wall = Quad(
#         Vec3(2, -2, -2),
#         Vec3(0, 4, 0),
#         Vec3(0, 0, -4),
#         Solid_Color(Vec3(0.4, 0.4, 0.4))
#     )
#
#     floor = Quad(
#         Vec3(-2, -2, -2),
#         Vec3(4, 0, 0),
#         Vec3(0, 0, -4),
#         Checker_Texture(30, Vec3(0.3, 0.3, 0.3), Vec3(0.7, 0.7, 0.7))
#     )
#
#     ceiling = Quad(
#         Vec3(-2, 2, -6),
#         Vec3(4, 0, 0),
#         Vec3(0, 0, 4),
#         Solid_Color(Vec3(0.7, 0.7, 0.7))
#     )
#
#     world.add(left_wall)
#     world.add(back_wall)
#     world.add(right_wall)
#     world.add(floor)
#     world.add(ceiling)
#
#     # =========================================================
#     # 2. STRUCTURED LANDMARKS (GRID ON FLOATING PLANE)
#     # =========================================================
#
#     landmarks = []
#
#     z_plane = -4.5
#
#     grid_size = 4
#     spacing = 0.4
#
#     colors = [
#         Vec3(1, 0, 0),
#         Vec3(0, 1, 0),
#         Vec3(0, 0, 1),
#         Vec3(1, 1, 0),
#         Vec3(1, 0, 1),
#         Vec3(0, 1, 1),
#     ]
#
#     idx = 0
#     for i in range(grid_size):
#         for j in range(grid_size):
#             x = (i - grid_size/2) * spacing
#             y = (j - grid_size/2) * spacing
#             z = z_plane + 0.2 * np.sin(i * j)  # slight depth variation
#
#             pos = Vec3(x, y, z)
#             col = Solid_Color(Vec3(0.2, 0.2, 0.2))
#
#             world.add(Sphere(pos, 0.08, col))
#             landmarks.append((pos, col))
#             idx += 1
#
#     # =========================================================
#     # 3. FLOATING OBJECTS (NON-COPLANAR STRUCTURE)
#     # =========================================================
#
#     extra_points = [
#         Vec3(-0.8, 0.8, -3.8),
#         Vec3(0.8, 0.7, -4.2),
#         Vec3(0.0, -0.8, -4.0),
#         Vec3(-0.6, -0.4, -3.6),
#         Vec3(0.6, -0.6, -4.6),
#     ]
#
#     for i, p in enumerate(extra_points):
#         world.add(
#             Sphere(p, 0.12, Solid_Color(Vec3(0.9, 0.9, 0.9)))
#         )
#         landmarks.append((p, Vec3(1,1,1)))
#
#     # =========================================================
#     # 4. AXIS LINES (for vanishing point demo)
#     # Three pairs of parallel lines along X, Y, Z axes
#     # centered in the scene, thin enough to look like lines
#     # =========================================================
#
#     axis_thickness = 0.3
#     axis_length = 40
#     axis_center = Vec3(0, 0, -4.25)
#
#     # X axis: two parallel horizontal lines (different y offsets)
#     for z_offset in [-0.5, 0.5]:
#         world.add(Quad(
#             Vec3(axis_center.x - axis_length, axis_center.y, axis_center.z + z_offset),
#             Vec3(2 * axis_length, 0, 0),  # direction: X
#             Vec3(axis_thickness, 0.0, axis_thickness),  # thickness: Y
#             Solid_Color(Vec3(1.0, 0.0, 0.0))  # red
#         ))
#
#     # Y axis: two parallel vertical lines (different x offsets)
#     for x_offset in [-0.5, 0.5]:
#         world.add(Quad(
#             Vec3(axis_center.x + x_offset, axis_center.y - axis_length, axis_center.z),
#             Vec3(0, 2 * axis_length, 0),  # direction: Y
#             Vec3(axis_thickness, 0.0, axis_thickness),  # thickness: X
#             Solid_Color(Vec3(0.0, 1.0, 0.0))  # green
#         ))
#
#         # Z axis: offset in X
#         world.add(Quad(
#             Vec3(axis_center.x + x_offset, axis_center.y, axis_center.z - axis_length),
#             Vec3(0, 0, 2 * axis_length),
#             Vec3(axis_thickness, 0.0, axis_thickness),
#             Solid_Color(Vec3(0.0, 0.0, 1.0))
#         ))
#
#
#     # =========================================================
#     # 5. CAMERA ORBIT (GOOD PARALLAX)
#     # =========================================================
#
#     center = Vec3(0, 0, -4)
#     elev = np.radians(60)
#     for i in range(N):
#         theta = (i / N) * 2 * np.pi
#
#         lookfrom = Vec3(
#             center.x + 10 * np.cos(theta),
#             center.y + 10 * np.sin(elev),  # varied elevation
#             center.z + 10 * np.sin(theta),
#         )
#
#         cam = Camera(
#             image_width=800,
#             samples_per_pixel=50,
#             max_depth=50,
#             reflectance=0.7,
#             vfov=70,
#             lookfrom=lookfrom,
#             lookat=center,
#             vup=Vec3(0, 1, 0),
#         )
#
#         render(world, cam)
#
#         with open(f'outputs/angles/img/{i+1}.ppm', 'wb') as f:
#             f.write(f'P6\n{cam.image_width} {cam.image_height}\n255\n'.encode())
#             f.write(cam.pixel_data.tobytes())
#
#         cam.recover_P()
#
#         np.save(f'outputs/angles/P/P_{i+1}.npy', cam.P)
#         np.save(f'outputs/angles/O/O_{i+1}.npy', cam.O)
#         np.save(f'outputs/angles/K/K_{i+1}.npy', cam.K)
#         np.save(f'outputs/angles/R/R_{i+1}.npy', cam.R)
#         np.save(f'outputs/angles/t/t_{i+1}.npy', cam.t)
#
#     return landmarks

def generate_N_views(N):
    world = World()

    # =========================================================
    # 1. CORNELL-STYLE BOX
    # =========================================================

    world.add(Quad(Vec3(-2, -2, -6), Vec3(0, 0, 4),  Vec3(0, 4, 0),  Solid_Color(Vec3(0.5, 0.5, 0.5))))
    world.add(Quad(Vec3(-2, -2, -6), Vec3(4, 0, 0),  Vec3(0, 4, 0),  Solid_Color(Vec3(0.6, 0.6, 0.6))))
    world.add(Quad(Vec3( 2, -2, -2), Vec3(0, 4, 0),  Vec3(0, 0, -4), Solid_Color(Vec3(0.4, 0.4, 0.4))))
    world.add(Quad(Vec3(-2, -2, -2), Vec3(4, 0, 0),  Vec3(0, 0, -4), Checker_Texture(30, Vec3(0.3, 0.3, 0.3), Vec3(0.7, 0.7, 0.7))))
    world.add(Quad(Vec3(-2,  2, -6), Vec3(4, 0, 0),  Vec3(0, 0, 4),  Solid_Color(Vec3(0.7, 0.7, 0.7))))

    # =========================================================
    # 2. STRUCTURED LANDMARKS (all gray — no colored spheres
    #    that would pollute the color masks)
    # =========================================================

    landmarks = []
    z_plane   = -4.5
    grid_size = 4
    spacing   = 0.4

    for i in range(grid_size):
        for j in range(grid_size):
            x   = (i - grid_size / 2) * spacing
            y   = (j - grid_size / 2) * spacing
            z   = z_plane + 0.2 * np.sin(i * j)
            pos = Vec3(x, y, z)
            world.add(Sphere(pos, 0.08, Solid_Color(Vec3(0.2, 0.2, 0.2))))
            landmarks.append((pos, Vec3(0.2, 0.2, 0.2)))

    # =========================================================
    # 3. FLOATING OBJECTS
    # =========================================================

    for p in [Vec3(-0.8, 0.8, -3.8), Vec3(0.8, 0.7, -4.2),
              Vec3(0.0, -0.8, -4.0), Vec3(-0.6, -0.4, -3.6),
              Vec3(0.6, -0.6, -4.6)]:
        world.add(Sphere(p, 0.12, Solid_Color(Vec3(0.9, 0.9, 0.9))))
        landmarks.append((p, Vec3(1, 1, 1)))

    # =========================================================
    # 4. CALIBRATION BOX — rotated 45° around Y so all three
    #    edge directions are diagonal in the image, pushing all
    #    three vanishing points well inside the frame.
    #    Edges are sphere chains — always visible regardless of
    #    camera angle, no quad orientation issues.
    # =========================================================

    angle = np.radians(45)
    R_box = np.array([[ np.cos(angle), 0, np.sin(angle)],
                      [ 0,             1, 0            ],
                      [-np.sin(angle), 0, np.cos(angle)]])

    bx, by, bz = 2.0, 1.5, 2.0  # bigger box — more pixels in image
    box_center = np.array([0.0, 0.0, -4.0])

    corners = {}
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                local = np.array([sx * bx, sy * by, sz * bz])
                wp    = R_box @ local + box_center
                corners[(sx, sy, sz)] = Vec3(*wp)

    def add_edge(p, q, color, n_spheres=30, radius=0.12):
        for k in range(n_spheres + 1):
            t   = k / n_spheres
            pos = Vec3(
                p.x + t * (q.x - p.x),
                p.y + t * (q.y - p.y),
                p.z + t * (q.z - p.z),
            )
            world.add(Sphere(pos, radius, Solid_Color(color)))

    # X-direction edges (red) — 4 edges
    for sy in [-1, 1]:
        for sz in [-1, 1]:
            add_edge(corners[(-1, sy, sz)], corners[(1, sy, sz)], Vec3(1.0, 0.0, 0.0))

    # Y-direction edges (green) — 4 edges
    for sx in [-1, 1]:
        for sz in [-1, 1]:
            add_edge(corners[(sx, -1, sz)], corners[(sx, 1, sz)], Vec3(0.0, 1.0, 0.0))

    # Z-direction edges (blue) — 4 edges
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            add_edge(corners[(sx, sy, -1)], corners[(sx, sy, 1)], Vec3(0.0, 0.0, 1.0))

    # =========================================================
    # 5. CAMERA ORBIT
    #    30° elevation — high enough to see the top of the box,
    #    low enough that all three vps stay near/inside frame.
    #    35° theta offset so no view is axis-aligned.
    # =========================================================

    center = Vec3(0, 0, -4)
    elev   = np.radians(30)

    for i in range(N):
        theta    = (i / N) * 2 * np.pi + np.radians(35)
        lookfrom = Vec3(
            center.x + 10 * np.cos(theta),
            center.y + 10 * np.sin(elev),
            center.z + 10 * np.sin(theta),
        )

        cam = Camera(
            image_width=1600,
            samples_per_pixel=50,
            max_depth=50,
            reflectance=0.7,
            vfov=70,
            lookfrom=lookfrom,
            lookat=center,
            vup=Vec3(0, 1, 0),
        )

        render(world, cam, i)

        with open(f'outputs/angles/img/{i+1}.ppm', 'wb') as f:
            f.write(f'P6\n{cam.image_width} {cam.image_height}\n255\n'.encode())
            f.write(cam.pixel_data.tobytes())

        cam.recover_P()

        np.save(f'outputs/angles/P/P_{i+1}.npy', cam.P)
        np.save(f'outputs/angles/O/O_{i+1}.npy', cam.O)
        np.save(f'outputs/angles/K/K_{i+1}.npy', cam.K)
        np.save(f'outputs/angles/R/R_{i+1}.npy', cam.R)
        np.save(f'outputs/angles/t/t_{i+1}.npy', cam.t)

    return landmarks


def main():
    # Initialize world
    world = World()

    scene = 6
    match scene:
        # Scene with ball on top of ground (ground is a quad)
        case 0:
            ground_color = Checker_Texture(250, Vec3(0.2, 0.3, 0.1), Vec3(0.9, 0.9, 0.9))
            ground = Quad(Vec3(-100, -1, -100), Vec3(200, 0, 0), Vec3(0, 0, 200), ground_color)
            world.add(ground)

            ball = Sphere(Vec3(0, 0, -1), 0.5, Solid_Color(Vec3(0.1, 0.2, 0.5)))
            world.add(ball)

        # Cornell box
        case 1:
            left_red     = Solid_Color(Vec3(1.0, 0.2, 0.2))
            back_green   = Solid_Color(Vec3(0.2, 1.0, 0.2))
            right_blue   = Solid_Color(Vec3(0.2, 0.2, 1.0))
            upper_orange = Solid_Color(Vec3(1.0, 0.5, 0.0))
            lower_teal   = Solid_Color(Vec3(0.2, 0.8, 0.8))

            left_red     = Quad(Vec3(-2, -2, -5), Vec3(0, 0,  4), Vec3(0, 4,  0), left_red)
            back_green   = Quad(Vec3(-2, -2, -5), Vec3(4, 0,  0), Vec3(0, 4,  0), back_green)
            right_blue   = Quad(Vec3( 2, -2, -1), Vec3(0, 4,  0), Vec3(0, 0, -4), right_blue)
            upper_orange = Quad(Vec3(-2,  2, -5), Vec3(4, 0,  0), Vec3(0, 0,  4), upper_orange)
            lower_teal   = Quad(Vec3(-2, -2, -1), Vec3(4, 0,  0), Vec3(0, 0, -4), lower_teal)

            world.add(left_red)
            world.add(back_green)
            world.add(right_blue)
            world.add(upper_orange)
            world.add(lower_teal)

        # Quadric floating in space
        case 2:
            desired_quadric = 'hyperboloid'
            center = (0, 0, -3)
            c_arr = np.array(center) # needed for matrix math below

            if desired_quadric == 'sphere':
                radius = 1
                A = np.eye(3)
                b = -2 * c_arr
                c = c_arr @ c_arr - radius**2
            elif desired_quadric == 'ellipsoid':
                A = np.array([[1, 0, 0],
                              [0, 1/4, 0],
                              [0, 0, 1]])
                b = -2 * A @ c_arr
                c = c_arr @ A @ c_arr - 1
            elif desired_quadric == 'cylinder':
                A = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
                b = -2 * A @ c_arr
                c = c_arr @ A @ c_arr - 1
            elif desired_quadric == 'hyperboloid':
                A = np.array([[ 1,  0, 0],
                              [ 0, -1, 0],
                              [ 0,  0, 1]])
                b = -2 * A @ c_arr
                c = c_arr @ A @ c_arr - 1
            elif desired_quadric == 'paraboloid':  # TODO
                pass
            else:
                raise NotImplementedError('Invalid quadric')

            quadric = Quadric(A, b, c, Solid_Color(Vec3(0.1, 0.2, 0.5)))
            world.add(quadric)

        # 2 parallel lines intersecting at the line at infinity
        case 3:
            p1 = Vec3(-1.5, -1, 5)
            p2 = Vec3(1.5, -1, 5)
            u = Vec3(0, 0, -100)  # line direction
            v = Vec3(0, 0.075, 0)  # thickness upward

            world.add(Quad(p1, u, v, Solid_Color(Vec3(0.8, 0.1, 0.1))))
            world.add(Quad(p2, u, v, Solid_Color(Vec3(0.1, 0.1, 0.8))))

            ground_color = Checker_Texture(250, Vec3(0.2, 0.3, 0.1), Vec3(0.9, 0.9, 0.9))
            ground = Quad(Vec3(-100, -0.999, -100), Vec3(200, 0, 0), Vec3(0, 0, 200), ground_color)
            world.add(ground)

        # Cross ratio confirmation
        case 4:
            ts_world = [-1.0, 0.0, 0.5, 1.0]
            line_orig = Vec3(0, 0, -1.5)
            line_dir = Vec3(1, 0, -0.5)

            # p(t) for each parameter
            #   t=-1 → (-1,  0, -1.0)   closest,  leftmost
            #   t= 0 → ( 0,  0, -1.5)
            #   t=0.5→ ( 0.5,0, -1.75)
            #   t= 1 → ( 1,  0, -2.0)   farthest, rightmost

            colors = [
                Vec3(1.0, 0.2, 0.2),  # red
                Vec3(0.2, 1.0, 0.2),  # green
                Vec3(0.2, 0.4, 1.0),  # blue
                Vec3(1.0, 0.85, 0.1),  # yellow
            ]
            # add all the points on the line
            points_3d = [line_orig + line_dir * t for t in ts_world]
            for pt, col in zip(points_3d, colors):
                world.add(Sphere(pt, 0.07, Solid_Color(col)))

            # Thin ribbon along the line so collinearity is visible in render
            ribbon_start = line_orig + line_dir * (-10.0) - Vec3(0, 0.015, 0)
            ribbon_u = line_dir * 20.0  # direction × total length, no subtraction needed
            ribbon_v = Vec3(0, 0.03, 0)
            world.add(Quad(ribbon_start, ribbon_u, ribbon_v,
                           Solid_Color(Vec3(0.1, 0.1, 0.1))))
            #
            # Ground for spatial reference
            ground = Quad(Vec3(-100, -0.5, -100), Vec3(200, 0, 0), Vec3(0, 0, 200),
                          Checker_Texture(250, Vec3(0.2, 0.3, 0.1), Vec3(0.9, 0.9, 0.9)))
            world.add(ground)

        # Quadric -> conic demonstration
        case 5:
            # render a unit sphere centered at [0, 0, -3]
            center = np.array([0, 0, -3])
            radius = 1
            A = np.eye(3)
            b = -2 * center
            c = center @ center - radius ** 2
            quadric = Quadric(A, b, c, Solid_Color(Vec3(0.1, 0.2, 0.5)))
            world.add(quadric)

            # Render tangent lines around sphere
            # this is our sample line at the top of the sphere horizontal. We rotate it around the sphere multiple times
            # and render each rotation as a new tangent line
            center = Vec3.from_array(center)
            num_tangents = 200
            for i in range(num_tangents):
                theta = (i / num_tangents) * 2 * np.pi
                t = (theta % (2 * np.pi)) / (2 * np.pi) # used to create color gradient
                p = Vec3(np.cos(theta), np.sin(theta), -3.0) * radius
                r = (p - center).normalize()
                u = Vec3(-np.sin(theta), np.cos(theta), 0).normalize() * 10
                v = r.cross(u).normalize() * 0.03
                color = Vec3(0.5 + 0.5 * np.cos(2 * np.pi * t),
                             0.5 + 0.5 * np.cos(2 * np.pi * (t + 1/3)),
                             0.5 + 0.5 * np.cos(2 * np.pi * (t + 2/3)))
                world.add(Quad(p, u, v, Solid_Color(color)))
                world.add(Quad(p, -u, v, Solid_Color(color)))

        # interesting scene to put in slides
        case 6:
            S = 0.5625  # horizontal squeeze factor = 225/400

            # ── FLOOR ──────────────────────────────────────────────────
            world.add(Quad(
                Vec3(-6 * S, -1, -2),
                Vec3(12 * S, 0, 0),
                Vec3(0, 0, -8),
                Checker_Texture(8, Vec3(0.85, 0.82, 0.75), Vec3(0.2, 0.18, 0.15))
            ))

            # ── CEILING ─────────────────────────────────────────────────
            world.add(Quad(
                Vec3(-6 * S, 3, -2),
                Vec3(12 * S, 0, 0),
                Vec3(0, 0, -8),
                Checker_Texture(5, Vec3(0.15, 0.15, 0.18), Vec3(0.08, 0.08, 0.10))
            ))

            # ── BACK WALL — three colored panels ───────────────────────
            world.add(Quad(
                Vec3(-6 * S, -1, -10),
                Vec3(4 * S, 0, 0), Vec3(0, 4, 0),
                Solid_Color(Vec3(0.12, 0.08, 0.25))
            ))
            world.add(Quad(
                Vec3(-2 * S, -1, -10),
                Vec3(4 * S, 0, 0), Vec3(0, 4, 0),
                Solid_Color(Vec3(0.08, 0.18, 0.28))
            ))
            world.add(Quad(
                Vec3(2 * S, -1, -10),
                Vec3(4 * S, 0, 0), Vec3(0, 4, 0),
                Solid_Color(Vec3(0.22, 0.08, 0.12))
            ))

            # ── LEFT WALL ───────────────────────────────────────────────
            world.add(Quad(
                Vec3(-6 * S, -1, -2),
                Vec3(0, 0, -8), Vec3(0, 4, 0),
                Checker_Texture(5, Vec3(0.4, 0.38, 0.33), Vec3(0.18, 0.17, 0.15))
            ))

            # ── RIGHT WALL ──────────────────────────────────────────────
            world.add(Quad(
                Vec3(6 * S, -1, -2),
                Vec3(0, 0, -8), Vec3(0, 4, 0),
                Checker_Texture(5, Vec3(0.4, 0.38, 0.33), Vec3(0.18, 0.17, 0.15))
            ))

            # ── CENTER ALTAR — wide flat ellipsoid ──────────────────────
            altar_c = np.array([0.0, -0.6, -6.0])
            A_a = np.diag([1 / (1.6 * S) ** 2, 1 / 0.5 ** 2, 1 / 1.0 ** 2])
            b_a = -2 * A_a @ altar_c
            c_a = altar_c @ A_a @ altar_c - 1
            world.add(Quadric(A_a, b_a, c_a, Solid_Color(Vec3(0.9, 0.75, 0.2))))

            # ── LEFT PILLAR — hyperboloid ───────────────────────────────
            pl_c = np.array([-2.8 * S, 0.5, -7.0])
            A_pl = np.diag([1 / (0.3 * S) ** 2, -1 / 2.0 ** 2, 1 / 0.3 ** 2])
            b_pl = -2 * A_pl @ pl_c
            c_pl = pl_c @ A_pl @ pl_c - 1
            world.add(Quadric(A_pl, b_pl, c_pl, Solid_Color(Vec3(0.45, 0.78, 0.88))))

            # ── RIGHT PILLAR — hyperboloid ──────────────────────────────
            pr_c = np.array([2.8 * S, 0.5, -7.0])
            A_pr = np.diag([1 / (0.3 * S) ** 2, -1 / 2.0 ** 2, 1 / 0.3 ** 2])
            b_pr = -2 * A_pr @ pr_c
            c_pr = pr_c @ A_pr @ pr_c - 1
            world.add(Quadric(A_pr, b_pr, c_pr, Solid_Color(Vec3(0.88, 0.45, 0.72))))

            # ── FLOATING ORB above altar ─────────────────────────────────
            world.add(Sphere(Vec3(0.0, 0.9, -6.0), 0.28,
                             Solid_Color(Vec3(0.95, 0.95, 1.0))))

            # ── SMALL ACCENT SPHERES around the orb ─────────────────────
            import math
            for k in range(6):
                angle = k * math.pi / 3
                ox = 0.65 * S * math.cos(angle)
                oz = -6.0 + 0.65 * math.sin(angle)
                world.add(Sphere(Vec3(ox, 0.75, oz), 0.1,
                                 Solid_Color(Vec3(
                                     0.5 + 0.5 * math.cos(angle),
                                     0.5 + 0.5 * math.cos(angle + 2.09),
                                     0.5 + 0.5 * math.cos(angle + 4.19)
                                 ))))

            # ── PEDESTAL under altar ─────────────────────────────────────
            world.add(Quad(
                Vec3(-0.7 * S, -1.0, -6.5),
                Vec3(1.4 * S, 0, 0), Vec3(0, 0, 1.0),
                Checker_Texture(3, Vec3(0.6, 0.58, 0.5), Vec3(0.25, 0.24, 0.2))
            ))
            for side_v, side_u in [
                (Vec3(-0.7 * S, -1, -5.5), Vec3(1.4 * S, 0, 0)),
                (Vec3(-0.7 * S, -1, -6.5), Vec3(0, 0, 1.0)),
                (Vec3(0.7 * S, -1, -6.5), Vec3(0, 0, 1.0)),
            ]:
                world.add(Quad(side_v, side_u, Vec3(0, 0.35, 0),
                               Solid_Color(Vec3(0.45, 0.43, 0.38))))

            # ── GEM SPHERES on floor ─────────────────────────────────────
            gems = [
                (Vec3(-1.8 * S, -0.87, -4.2), Vec3(0.9, 0.15, 0.15)),
                (Vec3(2.0 * S, -0.87, -4.5), Vec3(0.15, 0.75, 0.35)),
                (Vec3(-0.8 * S, -0.87, -8.8), Vec3(0.2, 0.3, 0.95)),
                (Vec3(1.1 * S, -0.87, -8.5), Vec3(0.85, 0.45, 0.05)),
                (Vec3(-3.5 * S, -0.87, -6.0), Vec3(0.7, 0.1, 0.8)),
                (Vec3(3.6 * S, -0.87, -5.8), Vec3(0.1, 0.8, 0.75)),
                (Vec3(0.3 * S, -0.87, -3.5), Vec3(0.95, 0.85, 0.1)),
            ]
            for pos, col in gems:
                world.add(Sphere(pos, 0.11, Solid_Color(col)))


        case _:
            raise NotImplementedError('No scene selected.')

    cam = Camera(image_width=1600, samples_per_pixel=100, max_depth=100, reflectance=0.7, vfov=60,
                 lookfrom=Vec3(0, 1, 3), lookat=Vec3(0, 1, -1), vup=Vec3(0, 1, 0))

    #cam = Camera(image_width=800, samples_per_pixel=100, max_depth=100, reflectance=0.7, vfov=90,
    #            lookfrom=Vec3(0, 0, 0), lookat=Vec3(0, 0, -1), vup=Vec3(0, 1, 0))
    render(world, cam, 1) # optimized parallel Numba render

    with open('outputs/output.ppm', 'wb') as f:
        f.write(f'P6\n{cam.image_width} {cam.image_height}\n255\n'.encode())
        f.write(cam.pixel_data.tobytes())


if __name__ == '__main__':
    #generate_N_views(8)
    main()
    #demo()
