import numpy as np
from numba import njit, prange
from vec3 import Vec3

# ── Data layout ───────────────────────────────────────────────────────────────
#
# Texture block (8 cols, appended to every primitive row):
#   [0]  type: 0=solid  1=checker
#   solid:   [1]=r  [2]=g  [3]=b
#   checker: [1]=scale  [2]=er  [3]=eg  [4]=eb  [5]=or  [6]=og  [7]=ob
#
# Sphere  (12 cols): cx cy cz radius                           | tex[8]
# Quad    (24 cols): Qx Qy Qz ux uy uz vx vy vz nx ny nz D wx wy wz | tex[8]
# Quadric (21 cols): A(9 flat) bx by bz c                     | tex[8]

TEX_COLS  = 8
SPH_COLS  = 4  + TEX_COLS   # 12
QUAD_COLS = 16 + TEX_COLS   # 24
QDR_COLS  = 13 + TEX_COLS   # 21


# ── Scene extraction (Python, runs once before render) ────────────────────────

def _v(v, i):
    """Get component i from either a Vec3 or numpy array."""
    return float(v[i]) if not hasattr(v, 'x') else (float(v.x), float(v.y), float(v.z))[i]


def _tex_block(texture):
    from texture import Checker_Texture
    block = np.zeros(TEX_COLS, dtype=np.float64)
    if isinstance(texture, Checker_Texture):
        block[0] = 1.0
        block[1] = float(texture.scale)
        e = texture.even.albedo
        o = texture.odd.albedo
        block[2] = e.x; block[3] = e.y; block[4] = e.z
        block[5] = o.x; block[6] = o.y; block[7] = o.z
    else:                                           # Solid_Color
        block[0] = 0.0
        c = texture.albedo
        block[1] = c.x; block[2] = c.y; block[3] = c.z
    return block


def extract_scene(world):
    from shape import Sphere, Quad, Quadric
    spheres  = []
    quads    = []
    quadrics = []

    for obj in world.objects:
        if isinstance(obj, Sphere):
            row = np.zeros(SPH_COLS, dtype=np.float64)
            row[0] = _v(obj.center, 0)
            row[1] = _v(obj.center, 1)
            row[2] = _v(obj.center, 2)
            row[3] = float(obj.radius)
            row[4:] = _tex_block(obj.texture)
            spheres.append(row)

        elif isinstance(obj, Quad):
            row = np.zeros(QUAD_COLS, dtype=np.float64)
            row[0]  = _v(obj.Q, 0);      row[1]  = _v(obj.Q, 1);      row[2]  = _v(obj.Q, 2)
            row[3]  = _v(obj.u, 0);      row[4]  = _v(obj.u, 1);      row[5]  = _v(obj.u, 2)
            row[6]  = _v(obj.v, 0);      row[7]  = _v(obj.v, 1);      row[8]  = _v(obj.v, 2)
            row[9]  = _v(obj.normal, 0); row[10] = _v(obj.normal, 1); row[11] = _v(obj.normal, 2)
            row[12] = float(obj.D)
            row[13] = _v(obj.w, 0);      row[14] = _v(obj.w, 1);      row[15] = _v(obj.w, 2)
            row[16:] = _tex_block(obj.texture)
            quads.append(row)

        elif isinstance(obj, Quadric):
            row = np.zeros(QDR_COLS, dtype=np.float64)
            for r in range(3):
                for c in range(3):
                    row[r*3 + c] = float(obj.A[r, c])
            row[9]  = float(obj.b[0])
            row[10] = float(obj.b[1])
            row[11] = float(obj.b[2])
            row[12] = float(obj.c)
            row[13:] = _tex_block(obj.texture)
            quadrics.append(row)

    def pack(lst, cols):
        return np.array(lst, dtype=np.float64) if lst else np.zeros((0, cols), dtype=np.float64)

    return pack(spheres, SPH_COLS), pack(quads, QUAD_COLS), pack(quadrics, QDR_COLS)


# ── Numba kernels (no Python objects past this point) ─────────────────────────

@njit
def dot3(ax, ay, az, bx, by, bz):
    return ax*bx + ay*by + az*bz


@njit
def eval_texture(tex, u, v):
    if tex[0] == 0:                     # solid
        return tex[1], tex[2], tex[3]
    s = int(np.floor(tex[1] * u))       # checker
    t = int(np.floor(tex[1] * v))
    if (s + t) % 2 == 0:
        return tex[2], tex[3], tex[4]
    return tex[5], tex[6], tex[7]


@njit
def lcg_next(state):
    state[0] = (state[0] * np.uint64(6364136223846793005)
                         + np.uint64(1442695040888963407))
    return state[0]


@njit
def random_float(state):
    return float(lcg_next(state) >> np.uint64(11)) / float(np.uint64(0x1FFFFFFFFFFFFF))


@njit
def random_unit_vector(state):
    while True:
        rx = random_float(state) * 2.0 - 1.0
        ry = random_float(state) * 2.0 - 1.0
        rz = random_float(state) * 2.0 - 1.0
        lensq = rx*rx + ry*ry + rz*rz
        if 1e-160 < lensq <= 1.0:
            inv = 1.0 / np.sqrt(lensq)
            return rx*inv, ry*inv, rz*inv


@njit
def hit_sphere(sph, ox, oy, oz, dx, dy, dz, t_min, t_max):
    cx, cy, cz, radius = sph[0], sph[1], sph[2], sph[3]
    ocx = cx-ox; ocy = cy-oy; ocz = cz-oz
    a    = dot3(dx,dy,dz,   dx,dy,dz)
    h    = dot3(dx,dy,dz,   ocx,ocy,ocz)
    c    = dot3(ocx,ocy,ocz, ocx,ocy,ocz) - radius*radius
    disc = h*h - a*c
    if disc < 0:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    sqrtd = np.sqrt(disc)
    t = (h - sqrtd) / a
    if t < t_min or t > t_max:
        t = (h + sqrtd) / a
        if t < t_min or t > t_max:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    px = ox+t*dx; py = oy+t*dy; pz = oz+t*dz
    inv_r = 1.0 / radius
    nx = (px-cx)*inv_r; ny = (py-cy)*inv_r; nz = (pz-cz)*inv_r
    theta = np.arccos(max(-1.0, min(1.0, -ny)))
    phi   = np.arctan2(-nz, nx) + np.pi
    u     = phi / (2.0 * np.pi)
    v     = theta / np.pi
    return True, t, nx, ny, nz, u, v


@njit
def hit_quad(quad, ox, oy, oz, dx, dy, dz, t_min, t_max):
    Qx,  Qy,  Qz  = quad[0],  quad[1],  quad[2]
    ux,  uy,  uz  = quad[3],  quad[4],  quad[5]
    vx,  vy,  vz  = quad[6],  quad[7],  quad[8]
    nx,  ny,  nz  = quad[9],  quad[10], quad[11]
    D             = quad[12]
    wx,  wy,  wz  = quad[13], quad[14], quad[15]

    denom = dot3(nx,ny,nz, dx,dy,dz)
    if abs(denom) < 1e-8:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    t = (D - dot3(nx,ny,nz, ox,oy,oz)) / denom
    if t < t_min or t > t_max:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    px = ox+t*dx; py = oy+t*dy; pz = oz+t*dz
    hx = px-Qx;  hy = py-Qy;  hz = pz-Qz

    # alpha = dot(w, cross(h, v))
    hcvx = hy*vz - hz*vy
    hcvy = hz*vx - hx*vz
    hcvz = hx*vy - hy*vx
    alpha = dot3(wx,wy,wz, hcvx,hcvy,hcvz)

    # beta = dot(w, cross(u, h))
    uchx = uy*hz - uz*hy
    uchy = uz*hx - ux*hz
    uchz = ux*hy - uy*hx
    beta = dot3(wx,wy,wz, uchx,uchy,uchz)

    if alpha < 0.0 or alpha > 1.0 or beta < 0.0 or beta > 1.0:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    return True, t, nx, ny, nz, alpha, beta


@njit
def hit_quadric(qdr, ox, oy, oz, dx, dy, dz, t_min, t_max):
    Adx = qdr[0]*dx + qdr[1]*dy + qdr[2]*dz
    Ady = qdr[3]*dx + qdr[4]*dy + qdr[5]*dz
    Adz = qdr[6]*dx + qdr[7]*dy + qdr[8]*dz
    Aox = qdr[0]*ox + qdr[1]*oy + qdr[2]*oz
    Aoy = qdr[3]*ox + qdr[4]*oy + qdr[5]*oz
    Aoz = qdr[6]*ox + qdr[7]*oy + qdr[8]*oz
    bx, by, bz = qdr[9], qdr[10], qdr[11]
    c_val      = qdr[12]

    a = dot3(dx,dy,dz,  Adx,Ady,Adz)
    b = 2.0 * dot3(ox,oy,oz, Adx,Ady,Adz) + dot3(bx,by,bz, dx,dy,dz)
    c = dot3(ox,oy,oz,  Aox,Aoy,Aoz) + dot3(bx,by,bz, ox,oy,oz) + c_val

    if abs(a) < 1e-8:
        if abs(b) < 1e-8:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        root = -c / b
        if root < t_min or root > t_max:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        disc = b*b - 4.0*a*c
        if disc < 0:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        sqrtd = np.sqrt(disc)
        root  = (-b - sqrtd) / (2.0*a)
        if root < t_min or root > t_max:
            root = (-b + sqrtd) / (2.0*a)
            if root < t_min or root > t_max:
                return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    px = ox+root*dx; py = oy+root*dy; pz = oz+root*dz
    nx = 2.0*(qdr[0]*px + qdr[1]*py + qdr[2]*pz) + bx
    ny = 2.0*(qdr[3]*px + qdr[4]*py + qdr[5]*pz) + by
    nz = 2.0*(qdr[6]*px + qdr[7]*py + qdr[8]*pz) + bz
    nlen = np.sqrt(dot3(nx,ny,nz, nx,ny,nz))
    return True, root, nx/nlen, ny/nlen, nz/nlen, 0.0, 0.0


@njit
def ray_color(ox, oy, oz, dx, dy, dz, max_depth,
              spheres, quads, quadrics, reflectance, state):
    tr, tg, tb = 1.0, 1.0, 1.0

    for _ in range(max_depth):
        closest_t  = 1e20
        hit_nx = hit_ny = hit_nz = 0.0
        hit_u  = hit_v  = 0.0
        hit_idx  = -1
        hit_type = -1

        for k in range(spheres.shape[0]):
            ok, t, nx, ny, nz, u, v = hit_sphere(
                spheres[k], ox, oy, oz, dx, dy, dz, 0.001, closest_t)
            if ok:
                closest_t = t
                hit_nx, hit_ny, hit_nz = nx, ny, nz
                hit_u, hit_v = u, v
                hit_idx  = k
                hit_type = 0

        for k in range(quads.shape[0]):
            ok, t, nx, ny, nz, u, v = hit_quad(
                quads[k], ox, oy, oz, dx, dy, dz, 0.001, closest_t)
            if ok:
                closest_t = t
                hit_nx, hit_ny, hit_nz = nx, ny, nz
                hit_u, hit_v = u, v
                hit_idx  = k
                hit_type = 1

        for k in range(quadrics.shape[0]):
            ok, t, nx, ny, nz, u, v = hit_quadric(
                quadrics[k], ox, oy, oz, dx, dy, dz, 0.001, closest_t)
            if ok:
                closest_t = t
                hit_nx, hit_ny, hit_nz = nx, ny, nz
                hit_u, hit_v = u, v
                hit_idx  = k
                hit_type = 2

        # no hit — sky gradient
        if hit_type < 0:
            inv = 1.0 / np.sqrt(dot3(dx,dy,dz, dx,dy,dz))
            a   = 0.5 * (dy * inv + 1.0)
            return tr*(1.0-0.5*a), tg*(1.0-0.3*a), tb*1.0

        # face-normal correction
        if dot3(dx,dy,dz, hit_nx,hit_ny,hit_nz) > 0:
            hit_nx, hit_ny, hit_nz = -hit_nx, -hit_ny, -hit_nz

        # texture lookup — tex block starts at col 4, 16, 13 respectively
        if hit_type == 0:
            cr, cg, cb = eval_texture(spheres[hit_idx,  4:], hit_u, hit_v)
        elif hit_type == 1:
            cr, cg, cb = eval_texture(quads[hit_idx,   16:], hit_u, hit_v)
        else:
            cr, cg, cb = eval_texture(quadrics[hit_idx, 13:], hit_u, hit_v)

        tr *= reflectance * cr
        tg *= reflectance * cg
        tb *= reflectance * cb

        # lambertian scatter
        ux, uy, uz = random_unit_vector(state)
        if dot3(ux,uy,uz, hit_nx,hit_ny,hit_nz) < 0:
            ux, uy, uz = -ux, -uy, -uz

        px = ox + closest_t*dx
        py = oy + closest_t*dy
        pz = oz + closest_t*dz
        ox = px + 1e-4*hit_nx
        oy = py + 1e-4*hit_ny
        oz = pz + 1e-4*hit_nz
        dx, dy, dz = ux, uy, uz

    return 0.0, 0.0, 0.0


@njit(parallel=True)
def render_kernel(pixel_data, spheres, quads, quadrics,
                  W, H, chunk_offset,
                  samples, max_depth, reflectance,
                  cam_ox, cam_oy, cam_oz,
                  p00x, p00y, p00z,
                  dux, duy, duz,
                  dvx, dvy, dvz):

    for i in prange(H):
        state = np.array([np.uint64(i + chunk_offset) * np.uint64(6364136223846793005)
                          + np.uint64(1)], dtype=np.uint64)
        for j in range(W):
            pr, pg, pb = 0.0, 0.0, 0.0
            for _ in range(samples):
                offU = random_float(state) - 0.5
                offV = random_float(state) - 0.5
                pdx  = p00x + (j + offU)*dux + (i + chunk_offset + offV)*dvx
                pdy  = p00y + (j + offU)*duy + (i + chunk_offset + offV)*dvy
                pdz  = p00z + (j + offU)*duz + (i + chunk_offset + offV)*dvz
                rdx  = pdx - cam_ox
                rdy  = pdy - cam_oy
                rdz  = pdz - cam_oz
                r, g, b = ray_color(
                    cam_ox, cam_oy, cam_oz,
                    rdx, rdy, rdz,
                    max_depth, spheres, quads, quadrics, reflectance, state)
                pr += r; pg += g; pb += b

            scale = 1.0 / samples
            pixel_data[i, j, 0] = min(int(256 * np.sqrt(max(pr*scale, 0.0))), 255)
            pixel_data[i, j, 1] = min(int(256 * np.sqrt(max(pg*scale, 0.0))), 255)
            pixel_data[i, j, 2] = min(int(256 * np.sqrt(max(pb*scale, 0.0))), 255)


# ── Public entry point ────────────────────────────────────────────────────────

def render(world, cam, num, chunk_rows=50):
    spheres, quads, quadrics = extract_scene(world)
    W, H = cam.image_width, cam.image_height
    cam.pixel_data = np.zeros((H, W, 3), dtype=np.uint8)

    print(f'Compiling Numba kernel: render #{num+1}')

    for row_start in range(0, H, chunk_rows):
        row_end    = min(row_start + chunk_rows, H)
        chunk_h    = row_end - row_start
        chunk_buf  = np.zeros((chunk_h, W, 3), dtype=np.uint8)

        render_kernel(
            chunk_buf, spheres, quads, quadrics,
            W, chunk_h, row_start,
            cam.samples_per_pixel, cam.max_depth, cam.reflectance,
            cam.pos.x,           cam.pos.y,           cam.pos.z,
            cam.pixel00_loc.x,   cam.pixel00_loc.y,   cam.pixel00_loc.z,
            cam.pixel_delta_u.x, cam.pixel_delta_u.y, cam.pixel_delta_u.z,
            cam.pixel_delta_v.x, cam.pixel_delta_v.y, cam.pixel_delta_v.z,
        )

        cam.pixel_data[row_start:row_end] = chunk_buf

        pct    = row_end / H * 100
        filled = int(pct * 0.4)
        bar    = '█' * filled + '░' * (40 - filled)
        print(f'\r[{bar}] {row_end}/{H} ({pct:.1f}%)', end='', flush=True)

    print('\nRender complete!')