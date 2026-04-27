import numpy as np
from ray import Ray
from shape import Hit_Record
from interval import Interval
from vec3 import Vec3


def sample_square():
    return np.random.rand(2) - 0.5


def normalize_color(pixel_color):
    intensity = Interval(0.0, 0.999)
    if pixel_color.x > 0:
        rbyte = int(256 * intensity.clamp(np.sqrt(pixel_color.x)))
    else: rbyte = 0
    if pixel_color.y > 0:
        gbyte = int(256 * intensity.clamp(np.sqrt(pixel_color.y)))
    else: gbyte = 0
    if pixel_color.z > 0:
        bbyte = int(256 * intensity.clamp(np.sqrt(pixel_color.z)))
    else: bbyte = 0
    return (rbyte, gbyte, bbyte)


def scatter(r_in, rec):
    direction = random_on_hemisphere(rec.normal)
    if direction.near_zero():
        direction = rec.normal
    scattered = Ray(rec.p + rec.normal * 1e-4, direction)
    attenuation = rec.texture.value(rec.u, rec.v, rec.p)
    return True, scattered, attenuation


def ray_color(r, depth, world, reflectance):
    if depth <= 0:
        return Vec3(0.0, 0.0, 0.0)

    rec = Hit_Record()
    if world.hit(r, Interval(0.001, np.inf), rec):
        hit, scattered, attenuation = scatter(r, rec)
        if hit:
            return ray_color(scattered, depth - 1, world, reflectance) * attenuation * reflectance
        return Vec3(0.0, 0.0, 0.0)

    # blended background color
    unit_direction = r.direction.normalize()
    a = 0.5 * (unit_direction.y + 1.0)
    return Vec3(1.0, 1.0, 1.0) * (1.0 - a) + Vec3(0.5, 0.7, 1.0) * a


def random_unit_vector():
    while True:
        p = Vec3(*(2.0 * np.random.rand(3) - 1.0))
        lensq = p.norm_sq()
        if 1e-160 < lensq <= 1:
            return p * (1.0 / np.sqrt(lensq))


def random_on_hemisphere(normal):
    on_unit_sphere = random_unit_vector()
    if on_unit_sphere.dot(normal) > 0:
        return on_unit_sphere
    else:
        return -on_unit_sphere


class Camera:
    def __init__(self, image_width, samples_per_pixel, max_depth, reflectance, vfov, lookfrom, lookat, vup):
        self.aspect_ratio = 16.0 / 9.0
        self.image_width = image_width
        self.samples_per_pixel = samples_per_pixel
        self.image_height = int(self.image_width / self.aspect_ratio)
        self.pixel_samples_scale = 1.0 / self.samples_per_pixel
        self.max_depth = max_depth
        self.reflectance = reflectance

        self.pos = lookfrom
        self.vfov = vfov
        self.lookat = lookat
        self.vup = vup

        # Viewport dimensions
        self.focal_length = (self.pos - self.lookat).norm()
        theta = np.deg2rad(self.vfov)
        h = np.tan(theta / 2)
        self.viewport_height = 2.0 * h * self.focal_length
        self.viewport_width = self.viewport_height * (self.image_width / self.image_height)

        # Basis vectors for camera coordinate frame
        self.w = (self.pos - self.lookat) * (1.0 / self.focal_length)
        self.u = self.vup.cross(self.w).normalize()
        self.v = self.w.cross(self.u)

        # Viewport edge vectors
        self.viewport_u = self.u * self.viewport_width
        self.viewport_v = self.v * -self.viewport_height

        # Per-pixel delta vectors
        self.pixel_delta_u = self.viewport_u * (1.0 / self.image_width)
        self.pixel_delta_v = self.viewport_v * (1.0 / self.image_height)

        # Upper-left pixel location
        self.viewport_upper_left = (self.pos
                                    - self.w * self.focal_length
                                    - self.viewport_u * 0.5
                                    - self.viewport_v * 0.5)
        self.pixel00_loc = self.viewport_upper_left + (self.pixel_delta_u + self.pixel_delta_v) * 0.5


    def render(self, world):
        self.pixel_data = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        for i in range(self.image_height):
            for j in range(self.image_width):
                pixel_color = Vec3(0.0, 0.0, 0.0)
                for _ in range(self.samples_per_pixel):
                    r = self.get_ray(i, j)
                    pixel_color += ray_color(r, self.max_depth, world, self.reflectance)
                self.pixel_data[i, j] = normalize_color(pixel_color * self.pixel_samples_scale)
            print(f'\rScanlines complete: {i + 1}/{self.image_height}', end='')
        print('\nImage render complete!')

    def get_ray(self, i, j):
        offset = sample_square()
        pixel_center = self.pixel00_loc + self.pixel_delta_u * j + self.pixel_delta_v * i
        pixel_sample = pixel_center + self.pixel_delta_u * offset[0] + self.pixel_delta_v * offset[1]
        return Ray(self.pos, pixel_sample - self.pos)


    ''' functions below are used to analyze the camera as a projective transformation:
        not actually used in the rendering'''

    def recover_P(self):
        # We recover the intrensic matrix, K
        self.cx = (self.image_width - 1) / 2.0  # principal point x
        self.cy = (self.image_height - 1) / 2.0  # principal point y
        f = self.focal_length * self.image_height / self.viewport_height  # focal length in *pixels*
        self.K = np.array([  # this is our K matrix for P = K[R|t]
            [f, 0, self.cx],
            [0, f, self.cy],
            [0, 0, 1],
        ], dtype=float)
        self.fx, self.fy = f, f

        # now recover extrensic matrix [R|t]
        lf = np.array(self.pos.to_array(), dtype=float)
        self.R = np.array([
            self.u.to_array(),  # row 0: right
            self.v.to_array(),  # row 1: up  (in camera frame, +Y is up)
            self.w.to_array(),  # row 2: -lookat
        ], dtype=float)

        self.t = -self.R @ lf  # translation vector (world origin in cam coords)
        self.O = lf # Camera centre in world coordinates (null space of P)

        # Full matrix: P = K[R|t}
        Rt = np.hstack([self.R, self.t[:, None]])   # 3×4
        self.P = self.K @ Rt
        self.O_h = np.append(lf, 1.0)  # 4-vector camera center

    def project(self, X_world: np.ndarray) -> np.ndarray:
        """
        Project a 3-D world point (or Nx3 array of points) into pixel
        coordinates using P.

        Parameters
        ----------
        X_world : array of shape (3,) or (N, 3)

        Returns
        -------
        pixels : array of shape (2,) or (N, 2)  — (col, row) = (x, y)
        """
        X = np.atleast_2d(X_world).T  # 3×N
        ones = np.ones((1, X.shape[1]))
        X_h = np.vstack([X, ones])  # 4×N homogeneous
        x_h = self.P @ X_h  # 3×N
        x_h /= x_h[2:3, :]  # normalise
        pixels = x_h[:2, :].T  # N×2  (col, row)
        return pixels.squeeze()

    def ray_direction(self, col: float, row: float) -> Vec3:
        """
        Return the (unnormalised) ray direction for pixel (col, row).
        Matches the geometry used by the renderer.
        """
        pixel_centre = (
                self.pixel00_loc
                + self.pixel_delta_u * col
                + self.pixel_delta_v * row
        )
        return pixel_centre - self.pos

    def unproject_to_ray(self, col: float, row: float):
        """
        Return (origin, direction) of the ray through pixel (col, row)
        as numpy arrays, using K⁻¹.

        This is the analytical inverse of project() — useful for
        verifying that project ∘ unproject = identity.
        """
        x_h = np.array([col, row, 1.0])
        d_cam = np.linalg.inv(self.K) @ x_h  # direction in camera frame
        d_world = self.R.T @ d_cam  # rotate to world frame
        return self.O.copy(), d_world / np.linalg.norm(d_world)

    def epipole_in(self, other: "Camera") -> np.ndarray:
        """
        Return the epipole of `other`'s camera centre projected into
        this camera's image plane.

        e = P @ C_other_h
        """
        e = self.P @ other.O_h
        return e / e[2]  # normalise to (u, v, 1)

    @staticmethod
    def fundamental_matrix(cam1: "Camera", cam2: "Camera") -> np.ndarray:
        """
        Compute the fundamental matrix F such that  x2^T F x1 = 0
        for corresponding points x1 (in cam1) and x2 (in cam2).

        Uses the formula:
            e2 = cam2.project(cam1.C)          ← epipole in image 2
            F  = [e2]× @ cam2.P @ pinv(cam1.P)

        Returns
        -------
        F : (3, 3) numpy array,  rank 2,  defined up to scale.
        """
        # Epipole in image 2: projection of cam1's centre into cam2
        e2 = cam2.P @ cam1.O_h  # 3-vector (homogeneous)
        e2 = e2 / e2[2]

        # Skew-symmetric matrix [e2]×
        e = e2
        E_cross = np.array([
            [0, -e[2], e[1]],
            [e[2], 0, -e[0]],
            [-e[1], e[0], 0],
        ])

        # F = [e2]× P2 P1⁺
        P1_plus = np.linalg.pinv(cam1.P)
        F = E_cross @ cam2.P @ P1_plus

        # Enforce rank-2 constraint
        U, s, Vt = np.linalg.svd(F)
        s[2] = 0.0
        F = U @ np.diag(s) @ Vt

        # Normalise so Frobenius norm = 1
        F /= np.linalg.norm(F)
        return F

    @staticmethod
    def verify_epipolar(
            F: np.ndarray,
            pts1: np.ndarray,
            pts2: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the epipolar residual  |x2^T F x1|  for each
        correspondence.  Should be ≈ 0 for correct matches.

        Parameters
        ----------
        pts1, pts2 : (N, 2) arrays of pixel coordinates
        """
        ones = np.ones((len(pts1), 1))
        x1 = np.hstack([pts1, ones])  # N×3
        x2 = np.hstack([pts2, ones])  # N×3
        residuals = np.abs(np.sum(x2 * (F @ x1.T).T, axis=1))
        return residuals

