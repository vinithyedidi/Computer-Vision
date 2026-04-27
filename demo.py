"""
demo.py  —  Vanishing Points → IAC (ω) → K → P
Pedagogical demonstration of camera calibration from colored axis lines.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from parallel import load_ppm, extract_colored_lines


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

IMG_DIR = "outputs/angles/img"
K_DIR   = "outputs/angles/K"
R_DIR   = "outputs/angles/R"
T_DIR   = "outputs/angles/t"
N_IMGS  = 8

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — VANISHING POINTS
# ─────────────────────────────────────────────────────────────────────────────

def vanishing_point(ppm_path, color_channel):
    result = extract_colored_lines(ppm_path, color_channel)
    if result is None:
        raise RuntimeError(f"Could not find lines for channel {color_channel}")
    l1, l2 = result
    vp = np.cross(l1, l2)
    return vp / np.linalg.norm(vp)


def triangle_area(vp_x, vp_y, vp_z):
    """Area of the triangle formed by the three vanishing points in pixel space."""
    def to_px(v):
        return np.array([v[0] / v[2], v[1] / v[2]])
    px, py, pz = to_px(vp_x), to_px(vp_y), to_px(vp_z)
    return 0.5 * abs((py - px)[0] * (pz - px)[1] - (pz - px)[0] * (py - px)[1])


def pick_best_image():
    from scipy import ndimage as ndi
    print("Scanning all images for best-conditioned vanishing point triangle...")
    best_img, best_score = 1, -1.0

    for i in range(1, N_IMGS + 1):
        path = f"{IMG_DIR}/{i}.ppm"
        try:
            img_i = load_ppm(path)
            H, W = img_i.shape[:2]

            vp_x = vanishing_point(path, 0)
            vp_y = vanishing_point(path, 1)
            vp_z = vanishing_point(path, 2)

            def to_px(v):
                return np.array([v[0]/v[2], v[1]/v[2]])

            px, py, pz = to_px(vp_x), to_px(vp_y), to_px(vp_z)

            # skip if any vp is more than 1.5x image size outside frame
            margin = 1.5
            def in_bounds(p):
                return (-margin*W < p[0] < (1+margin)*W and
                        -margin*H < p[1] < (1+margin)*H)

            if not all(in_bounds(p) for p in [px, py, pz]):
                print(f"  img {i}: SKIP — vp outside bounds  {[p.tolist() for p in [px,py,pz]]}")
                continue

            # compute the actual constraint matrix condition number
            def row(vi, vj):
                return np.array([
                    vi[0]*vj[0] + vi[1]*vj[1],
                    vi[0]*vj[2] + vi[2]*vj[0],
                    vi[1]*vj[2] + vi[2]*vj[1],
                    vi[2]*vj[2]
                ])
            A = np.array([row(vp_x, vp_y), row(vp_x, vp_z), row(vp_y, vp_z)])
            cond = np.linalg.cond(A)

            # score = 1/cond — lower condition = better
            score = 1.0 / cond
            area = triangle_area(vp_x, vp_y, vp_z)
            print(f"  img {i}: cond={cond:>12.1f}  area={area:>10.0f}  score={score:.2e}")

            if score > best_score:
                best_score = score
                best_img = i

        except Exception as e:
            print(f"  img {i}: FAILED — {e}")

    print(f"\n  → Best image: {best_img}  (score = {best_score:.2e})\n")
    return best_img

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — IAC  ω = (K Kᵀ)⁻¹
# ─────────────────────────────────────────────────────────────────────────────

def compute_omega(v1, v2, v3):
    """
    Full parameterization of ω (square pixel, zero skew, 4 unknowns):
        ω = [[w1,  0,  w2],
             [0,   w1, w3],
             [w2,  w3, w4]]
    where w1 = 1/f², w2 = -cx/f², w3 = -cy/f², w4 = (cx²+cy²)/f² + 1

    Each pair viᵀ ω vj = 0 gives one equation in [w1, w2, w3, w4].
    3 pairs → underdetermined (4 unknowns) → solve via SVD (least-norm).
    """
    def row(vi, vj):
        # viᵀ ω vj expanded:
        # (vi[0]*vj[0] + vi[1]*vj[1]) * w1
        # + (vi[0]*vj[2] + vi[2]*vj[0]) * w2
        # + (vi[1]*vj[2] + vi[2]*vj[1]) * w3
        # + vi[2]*vj[2] * w4
        return np.array([
            vi[0]*vj[0] + vi[1]*vj[1],   # w1
            vi[0]*vj[2] + vi[2]*vj[0],   # w2
            vi[1]*vj[2] + vi[2]*vj[1],   # w3
            vi[2]*vj[2]                   # w4
        ])

    A = np.array([row(v1, v2), row(v1, v3), row(v2, v3)])
    print(f"  Constraint matrix condition number: {np.linalg.cond(A):.1f}")

    # SVD least-norm solution (null space of A)
    _, _, Vt = np.linalg.svd(A)
    w1, w2, w3, w4 = Vt[-1]

    # enforce w1 > 0 (since w1 = 1/f² must be positive)
    if w1 < 0:
        w1, w2, w3, w4 = -w1, -w2, -w3, -w4

    omega = np.array([[w1,  0,  w2],
                      [0,   w1, w3],
                      [w2,  w3, w4]])
    return omega

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — K from ω
# ─────────────────────────────────────────────────────────────────────────────

def recover_K(omega):
    """
    Extract K directly from ω = (K Kᵀ)⁻¹ algebraically.
    For zero-skew, square-pixel camera:
        ω = [[w1,  0,  w2],
             [0,   w1, w3],
             [w2,  w3, w4]]
    where:
        w1 = 1/f²
        w2 = -cx/f²
        w3 = -cy/f²
        w4 = (cx² + cy²)/f² + 1

    Solving directly:
        f  = 1/sqrt(w1)
        cx = -w2/w1
        cy = -w3/w1
    """
    w1 = omega[0, 0]
    w2 = omega[0, 2]
    w3 = omega[1, 2]

    if w1 <= 0:
        raise RuntimeError(f"ω[0,0] = {w1} ≤ 0 — invalid, check vanishing points.")

    f  = 1.0 / np.sqrt(w1)
    cx = -w2 / w1
    cy = -w3 / w1

    K = np.array([[f,  0,  cx],
                  [0,  f,  cy],
                  [0,  0,  1.0]])
    return K

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — P = K [R | t]
# ─────────────────────────────────────────────────────────────────────────────

def recover_P(K, R, t):
    return K @ np.hstack([R, t.reshape(3, 1)])

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def draw_line(ax, l, W, H, **kw):
    a, b, c = l
    pts = []
    candidates = [
        (0,          -c/b           if abs(b) > 1e-9 else None),
        (W,  -(c + a*W)/b           if abs(b) > 1e-9 else None),
        (-c/a if abs(a) > 1e-9 else None,        0),
        (-(c + b*H)/a if abs(a) > 1e-9 else None, H),
    ]
    for x, y in candidates:
        if x is not None and y is not None and 0 <= x <= W and 0 <= y <= H:
            pts.append((x, y))
    if len(pts) >= 2:
        ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], **kw)


def plot_demo(ppm_path, vp_x, vp_y, vp_z, omega, K_rec, K_gt=None):
    img = load_ppm(ppm_path)
    H, W = img.shape[:2]

    axis_info = [
        (0, vp_x, 'red',  'vp_X (X axis)'),
        (1, vp_y, 'lime', 'vp_Y (Y axis)'),
        (2, vp_z, 'cyan', 'vp_Z (Z axis)'),
    ]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(img)

    # fitted axis lines
    for ch, vp, color, label in axis_info:
        result = extract_colored_lines(ppm_path, ch)
        if result:
            for l in result:
                draw_line(ax, l, W, H, color=color, lw=2, linestyle='--', alpha=0.8)

    # vanishing points
    for ch, vp, color, label in axis_info:
        if abs(vp[2]) > 1e-10:
            px, py = vp[0]/vp[2], vp[1]/vp[2]
            if -W < px < 2*W and -H < py < 2*H:
                ax.plot(px, py, 'o', color=color, ms=14,
                        markeredgecolor='white', markeredgewidth=2, zorder=6)
                ax.annotate(label, (px, py), xytext=(10, 8),
                            textcoords='offset points', color=color,
                            fontsize=11, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      facecolor='black', alpha=0.5))

    # horizon = join of vp_x and vp_z (both lie on the horizontal plane)
    horizon = np.cross(vp_x, vp_z)
    a, b, c = horizon
    if abs(b) > 1e-8:
        ax.plot([0, W], [-c/b, -(c + a*W)/b],
                color='yellow', lw=2.5, zorder=5)

    # principal point from recovered K
    cx, cy = K_rec[0, 2], K_rec[1, 2]
    ax.plot(cx, cy, '+', color='white', ms=22, markeredgewidth=3, zorder=7)
    ax.annotate(f'Principal pt\n({cx:.0f}, {cy:.0f})',
                (cx, cy), xytext=(15, -35), textcoords='offset points',
                color='white', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

    f_rec = K_rec[0, 0]
    title = [
        "Vanishing Points  →  IAC (ω)  →  K  →  P",
        f"Recovered:    f = {f_rec:.1f}   cx = {cx:.1f}   cy = {cy:.1f}",
    ]
    if K_gt is not None:
        title.append(
            f"Ground truth: f = {K_gt[0,0]:.1f}   cx = {K_gt[0,2]:.1f}   cy = {K_gt[1,2]:.1f}"
        )
    ax.set_title("\n".join(title), fontsize=11)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.legend(handles=[
        mpatches.Patch(color='red',    label='X-axis edges → vp_X'),
        mpatches.Patch(color='lime',   label='Y-axis edges → vp_Y'),
        mpatches.Patch(color='cyan',   label='Z-axis edges → vp_Z'),
        mpatches.Patch(color='yellow', label='Horizon = join(vp_X, vp_Z)'),
    ], loc='upper right', fontsize=10, facecolor='black', labelcolor='white', framealpha=0.7)

    plt.tight_layout()
    plt.savefig("outputs/demo/demo_result.png", dpi=150)
    plt.show()


def plot_horizon_demo(ppm_path, vp_x, vp_y, vp_z, omega, K_rec):
    """
    Demonstrates that the horizon line is the image of the plane at infinity.
    Shows:
      - The three vanishing points
      - The horizon line (join of vp_x and vp_z — both horizontal directions)
      - That vp_y (vertical) is NOT on the horizon
      - The recovered 3D directions from K⁻¹ v
    """
    img = load_ppm(ppm_path)
    H, W = img.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ── LEFT: image with vanishing points and horizon ─────────────────
    ax = axes[0]
    ax.imshow(img)

    def to_px(v):
        return np.array([v[0]/v[2], v[1]/v[2]])

    # draw fitted lines per axis
    axis_info = [
        (0, vp_x, 'red',  'vp_X'),
        (1, vp_y, 'lime', 'vp_Y'),
        (2, vp_z, 'cyan', 'vp_Z'),
    ]
    for ch, vp, color, label in axis_info:
        result = extract_colored_lines(ppm_path, ch)
        if result:
            for l in result:
                draw_line(ax, l, W, H, color=color, lw=2, linestyle='--', alpha=0.7)

    # vanishing points
    vp_pixels = {}
    for ch, vp, color, label in axis_info:
        px = to_px(vp)
        vp_pixels[label] = px
        # draw even if outside image
        ax.plot(px[0], px[1], 'o', color=color, ms=14,
                markeredgecolor='white', markeredgewidth=2, zorder=6)
        ax.annotate(label, px, xytext=(10, 8), textcoords='offset points',
                    color=color, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    # horizon line = join of vp_x and vp_z (both lie in the horizontal plane y=0)
    horizon = np.cross(vp_x, vp_z)
    a, b, c = horizon
    if abs(b) > 1e-8:
        y_left  = -c / b
        y_right = -(c + a * W) / b
        ax.plot([0, W], [y_left, y_right],
                color='yellow', lw=3, zorder=5, label='Horizon (image of Π∞)')

        # annotate the horizon
        ax.annotate('Horizon line\n= image of plane at infinity Π∞',
                    xy=(W*0.05, y_left + (y_right-y_left)*0.05),
                    color='yellow', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

    # show vp_y is off the horizon — drop a perpendicular to horizon line
    py_px = to_px(vp_y)
    # closest point on horizon to vp_y
    n = np.array([a, b])
    n = n / np.linalg.norm(n)
    if abs(b) > 1e-8:
        # parameterize horizon as p(t) = (t, (-c-a*t)/b)
        # find t minimizing distance to py_px
        t_close = (py_px[0] - a*(a*py_px[0] + b*py_px[1] + c)/(a**2+b**2))
        y_close = (-c - a*t_close) / b
        ax.annotate('', xy=(t_close, y_close), xytext=(py_px[0], py_px[1]),
                    arrowprops=dict(arrowstyle='<->', color='lime', lw=2))
        mid = np.array([(t_close+py_px[0])/2, (y_close+py_px[1])/2])
        ax.annotate('vp_Y not on horizon\n(Y axis is vertical, not horizontal)',
                    xy=mid, xytext=(mid[0]+40, mid[1]-40),
                    textcoords='data', color='lime', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6),
                    arrowprops=dict(arrowstyle='->', color='lime'))

    ax.set_xlim(-W*0.1, W*1.1)
    ax.set_ylim(H*1.1, -H*0.1)   # extra margin to show off-screen vps
    ax.set_title("Vanishing Points & Horizon Line", fontsize=13)
    ax.legend(handles=[
        mpatches.Patch(color='red',    label='X-axis lines → vp_X'),
        mpatches.Patch(color='lime',   label='Y-axis lines → vp_Y (vertical)'),
        mpatches.Patch(color='cyan',   label='Z-axis lines → vp_Z'),
        mpatches.Patch(color='yellow', label='Horizon = join(vp_X, vp_Z)'),
    ], loc='lower right', fontsize=9, facecolor='black', labelcolor='white', framealpha=0.8)

    # ── RIGHT: recovered 3D directions ────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('black')

    K_inv = np.linalg.inv(K_rec)

    dirs = {
        'X (red)':   (vp_x, 'red'),
        'Y (green)': (vp_y, 'lime'),
        'Z (blue)':  (vp_z, 'cyan'),
    }

    origin = np.zeros(3)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor('black')

    for name, (vp, color) in dirs.items():
        # back-project vanishing point to 3D direction
        d = K_inv @ vp
        d = d / np.linalg.norm(d)
        ax2.quiver(0, 0, 0, d[0], d[1], d[2],
                   color=color, linewidth=3, arrow_length_ratio=0.15,
                   label=f'{name}: [{d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f}]')

    # check orthogonality — print angles
    d_x = K_inv @ vp_x; d_x /= np.linalg.norm(d_x)
    d_y = K_inv @ vp_y; d_y /= np.linalg.norm(d_y)
    d_z = K_inv @ vp_z; d_z /= np.linalg.norm(d_z)
    ang_xy = np.degrees(np.arccos(np.clip(d_x @ d_y, -1, 1)))
    ang_yz = np.degrees(np.arccos(np.clip(d_y @ d_z, -1, 1)))
    ang_xz = np.degrees(np.arccos(np.clip(d_x @ d_z, -1, 1)))

    ax2.set_xlim(-1, 1); ax2.set_ylim(-1, 1); ax2.set_zlim(-1, 1)
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.set_title(
        f"Recovered 3D axis directions via K⁻¹v\n"
        f"∠(X,Y)={ang_xy:.1f}°  ∠(Y,Z)={ang_yz:.1f}°  ∠(X,Z)={ang_xz:.1f}°\n"
        f"(all should be 90°)",
        fontsize=11, color='white'
    )
    ax2.legend(fontsize=9, loc='upper left', facecolor='black', labelcolor='white')
    ax2.tick_params(colors='white')
    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    ax2.zaxis.label.set_color('white')

    plt.suptitle(
        "The Horizon Line is the Image of the Plane at Infinity Π∞\n"
        "Vanishing points on Π∞ encode 3D scene directions",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("outputs/demo/horizon_demo.png", dpi=150)
    plt.show()


def plot_angle_demo(ppm_path, vp_x, vp_y, vp_z, omega, K_rec):
    """
    Demonstrates that angles between 3D scene directions are recoverable
    from vanishing points and the IAC.

    Two ways to measure the angle between directions i and j:
      1. Via IAC:        cos(θ) = vi^T ω vj / sqrt((vi^T ω vi)(vj^T ω vj))
      2. Via K^{-1}:     cos(θ) = (K^{-1} vi) · (K^{-1} vj)  (after normalizing)

    Both give the same result. Ground truth is 90° since X, Y, Z are orthogonal.
    """
    img  = load_ppm(ppm_path)
    H, W = img.shape[:2]
    K_inv = np.linalg.inv(K_rec)

    vps   = {'X': (vp_x, 'red'), 'Y': (vp_y, 'lime'), 'Z': (vp_z, 'cyan')}
    pairs = [('X', 'Y'), ('Y', 'Z'), ('X', 'Z')]

    # ── compute angles both ways ───────────────────────────────────────
    results = []
    for na, nb in pairs:
        va, _ = vps[na]
        vb, _ = vps[nb]

        # method 1: via IAC
        num   = va @ omega @ vb
        denom = np.sqrt((va @ omega @ va) * (vb @ omega @ vb))
        cos_iac = np.clip(num / denom, -1, 1)
        ang_iac = np.degrees(np.arccos(cos_iac))

        # method 2: via K^{-1}
        da = K_inv @ va;  da /= np.linalg.norm(da)
        db = K_inv @ vb;  db /= np.linalg.norm(db)
        ang_kinv = np.degrees(np.arccos(np.clip(da @ db, -1, 1)))

        results.append((na, nb, ang_iac, ang_kinv, da, db))

    # ── figure: two panels ────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 7))
    fig.patch.set_facecolor('black')

    # ── LEFT: image annotated with vp pairs and angle arcs ────────────
    ax1 = fig.add_subplot(121)
    ax1.imshow(img)
    ax1.set_facecolor('black')

    pair_colors = ['orange', 'magenta', 'white']

    def to_px(v):
        return np.array([v[0]/v[2], v[1]/v[2]])

    # draw fitted lines
    for ch, (vp, color) in enumerate(vps.values()):
        result = extract_colored_lines(ppm_path, ch)
        if result:
            for l in result:
                draw_line(ax1, l, W, H, color=color, lw=2, linestyle='--', alpha=0.7)

    # draw vanishing points
    for name, (vp, color) in vps.items():
        px = to_px(vp)
        if -2*W < px[0] < 3*W and -2*H < px[1] < 3*H:
            ax1.plot(px[0], px[1], 'o', color=color, ms=14,
                     markeredgecolor='white', markeredgewidth=2, zorder=6)
            ax1.annotate(f'vp_{name}', px, xytext=(10, 8),
                         textcoords='offset points', color=color,
                         fontsize=11, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2',
                                   facecolor='black', alpha=0.6))

    # annotate each pair with recovered angle
    for k, ((na, nb, ang_iac, ang_kinv, _, _), pc) in enumerate(zip(results, pair_colors)):
        va_px = to_px(vps[na][0])
        vb_px = to_px(vps[nb][0])
        mid   = (va_px + vb_px) / 2
        # clamp annotation to inside image
        mid_clamped = np.clip(mid, [20, 20], [W-20, H-20])
        ax1.annotate(
            f'∠({na},{nb})\nIAC:  {ang_iac:.1f}°\nK⁻¹v: {ang_kinv:.1f}°\nGT:   90.0°',
            xy=mid_clamped,
            color=pc, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.75)
        )
        # draw line connecting the two vps (clipped)
        ax1.plot([va_px[0], vb_px[0]], [va_px[1], vb_px[1]],
                 color=pc, lw=1.5, linestyle=':', alpha=0.5, zorder=4)

    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)
    ax1.set_title("Vanishing Point Pairs → Scene Angles", color='white', fontsize=13)
    ax1.tick_params(colors='white')

    # ── RIGHT: 3D arrow plot with angles labeled ───────────────────────
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor('black')

    dir_vecs = {}
    for name, (vp, color) in vps.items():
        d = K_inv @ vp
        d = d / np.linalg.norm(d)
        dir_vecs[name] = d
        ax2.quiver(0, 0, 0, d[0], d[1], d[2],
                   color=color, linewidth=3, arrow_length_ratio=0.15)
        ax2.text(d[0]*1.15, d[1]*1.15, d[2]*1.15, f'{name}\n{d.round(2)}',
                 color=color, fontsize=9, ha='center')

    # draw arcs between each pair to visualize the angle
    for (na, nb, ang_iac, _, _, _), pc in zip(results, pair_colors):
        da = dir_vecs[na]
        db = dir_vecs[nb]
        # arc: interpolate between da and db on the unit sphere
        ts  = np.linspace(0, 1, 30)
        arc = np.array([(1-t)*da + t*db for t in ts])
        arc = arc / np.linalg.norm(arc, axis=1, keepdims=True) * 0.4
        ax2.plot(arc[:,0], arc[:,1], arc[:,2],
                 color=pc, lw=2, alpha=0.8)
        mid_arc = arc[15]
        ax2.text(mid_arc[0]*1.3, mid_arc[1]*1.3, mid_arc[2]*1.3,
                 f'{ang_iac:.1f}°', color=pc, fontsize=10, fontweight='bold')

    ax2.set_xlim(-1, 1); ax2.set_ylim(-1, 1); ax2.set_zlim(-1, 1)
    ax2.set_xlabel('X', color='white')
    ax2.set_ylabel('Y', color='white')
    ax2.set_zlabel('Z', color='white')
    ax2.tick_params(colors='white')
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.set_title("Recovered 3D Directions & Angles\n(via IAC and K⁻¹)",
                  color='white', fontsize=12)

    # ── summary table ──────────────────────────────────────────────────
    plt.suptitle(
        "Measuring 3D Angles from a Single Image via the IAC\n"
        r"$\cos\theta_{ij} = \frac{v_i^T \omega v_j}{\sqrt{(v_i^T \omega v_i)(v_j^T \omega v_j)}}$",
        fontsize=13, fontweight='bold', color='white'
    )

    print("\n" + "=" * 55)
    print("ANGLE RECOVERY FROM VANISHING POINTS")
    print("=" * 55)
    print(f"  {'Pair':<8} {'Via IAC':>10} {'Via K⁻¹v':>10} {'GT':>8}")
    print(f"  {'-'*40}")
    for na, nb, ang_iac, ang_kinv, _, _ in results:
        print(f"  ({na},{nb})    {ang_iac:>9.2f}°  {ang_kinv:>9.2f}°  {'90.00°':>8}")

    plt.tight_layout()
    plt.savefig("outputs/demo/angle_demo.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def demo():

    # ── pick the best-conditioned image automatically ──────────────────────
    img_num = pick_best_image()
    ppm_path = f"{IMG_DIR}/{img_num}.ppm"

    # ── DIAGNOSTIC: visualize raw color masks ──────────────────────────────
    img = load_ppm(ppm_path)
    r = img[:, :, 0].astype(float)
    g = img[:, :, 1].astype(float)
    b = img[:, :, 2].astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    masks = [
        ((r > 150) & (g < 80) & (b < 80), 'Red mask (X axis)', 'Reds'),
        ((g > 150) & (r < 80) & (b < 80), 'Green mask (Y axis)', 'Greens'),
        ((b > 150) & (r < 80) & (g < 80), 'Blue mask (Z axis)', 'Blues'),
    ]
    for ax, (mask, title, cmap) in zip(axes, masks):
        ax.imshow(mask, cmap=cmap)
        ax.set_title(f"{title}\n{mask.sum()} pixels")
    plt.tight_layout()
    plt.savefig("outputs/debug_masks.png", dpi=100)
    plt.show()
    input("Check debug_masks.png — press Enter to continue...")

    # ── Step 1: vanishing points ───────────────────────────────────────────
    print("=" * 55)
    print("STEP 1 — Vanishing Points")
    print("=" * 55)
    vp_x = vanishing_point(ppm_path, 0)
    vp_y = vanishing_point(ppm_path, 1)
    vp_z = vanishing_point(ppm_path, 2)

    def to_px(v): return np.array([v[0]/v[2], v[1]/v[2]])
    print(f"  vp_X pixel coords: {to_px(vp_x)}")
    print(f"  vp_Y pixel coords: {to_px(vp_y)}")
    print(f"  vp_Z pixel coords: {to_px(vp_z)}")
    print(f"  Triangle area:     {triangle_area(vp_x, vp_y, vp_z):.1f} px²")

    # ── Step 2: IAC ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 2 — IAC  ω = (K Kᵀ)⁻¹")
    print("=" * 55)
    omega = compute_omega(vp_x, vp_y, vp_z)
    print(f"  ω =\n{omega}")
    print("  Orthogonality checks (viᵀ ω vj, should be ≈ 0):")
    for na, va, nb, vb in [('X', vp_x, 'Y', vp_y),
                            ('Y', vp_y, 'Z', vp_z),
                            ('X', vp_x, 'Z', vp_z)]:
        print(f"    v{na}·ω·v{nb} = {va @ omega @ vb:.2e}")

    # ── Step 3: recover K ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 3 — Recover K via Cholesky on ω⁻¹")
    print("=" * 55)
    K_rec = recover_K(omega)
    print(f"  K_recovered =\n{K_rec}")

    K_gt = np.load(f"{K_DIR}/K_{img_num}.npy")
    print(f"\n  K_ground_truth =\n{K_gt}")
    print(f"\n  Error on f:  {abs(K_rec[0,0] - K_gt[0,0]):.4f}")
    print(f"  Error on cx: {abs(K_rec[0,2] - K_gt[0,2]):.4f}")
    print(f"  Error on cy: {abs(K_rec[1,2] - K_gt[1,2]):.4f}")

    # ── Step 4: recover P ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 4 — Recover P = K [R | t]")
    print("=" * 55)
    R = np.load(f"{R_DIR}/R_{img_num}.npy")
    t = np.load(f"{T_DIR}/t_{img_num}.npy")
    P_rec = recover_P(K_rec, R, t)
    P_gt  = recover_P(K_gt,  R, t)
    print(f"  P (recovered K) =\n{P_rec}")
    print(f"\n  P (ground truth K) =\n{P_gt}")

    # ── Step 5: visualize ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 5 — Visualization")
    print("=" * 55)
    plot_demo(ppm_path, vp_x, vp_y, vp_z, omega, K_rec, K_gt)

    print("\n" + "=" * 55)
    print("STEP 6 — Horizon Line & Scene Geometry")
    print("=" * 55)
    plot_horizon_demo(ppm_path, vp_x, vp_y, vp_z, omega, K_rec)

    print("\n" + "=" * 55)
    print("STEP 7 — Angle Recovery from Vanishing Points")
    print("=" * 55)
    plot_angle_demo(ppm_path, vp_x, vp_y, vp_z, omega, K_rec)