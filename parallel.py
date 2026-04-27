import numpy as np
from scipy import ndimage


def load_ppm(path):
    with open(path, 'rb') as f:
        magic = f.readline().strip()
        assert magic == b'P6', f"Expected P6 PPM, got {magic}"
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        W, H = map(int, line.split())
        maxval = int(f.readline().strip())
        assert maxval == 255
        data = np.frombuffer(f.read(), dtype=np.uint8)
        img = data.reshape(H, W, 3)
    return img


def fit_line_to_points(pts):
    """Fit homogeneous line [a,b,c] to (N,2) pixel coords via SVD."""
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    _, _, Vt = np.linalg.svd(pts_h)
    return Vt[-1]


def extract_colored_lines(ppm_path, color_channel):
    """
    Segment pixels by color, keep the two largest connected components
    (filters out small sphere blobs), fit a line to each.

    Returns (l1, l2) homogeneous lines, or None on failure.
    """
    img = load_ppm(ppm_path)
    r = img[:, :, 0].astype(float)
    g = img[:, :, 1].astype(float)
    b = img[:, :, 2].astype(float)

    if color_channel == 0:  # red
        mask = (r > 150) & (g < 50) & (b < 50)
    elif color_channel == 1:  # green
        mask = (g > 150) & (r < 50) & (b < 50)
    elif color_channel == 2:  # blue
        mask = (b > 150) & (r < 50) & (g < 50)
    else:
        raise ValueError("color_channel must be 0, 1, or 2")

    labeled, num_features = ndimage.label(mask)

    if num_features < 2:
        print(f"  Warning: only {num_features} connected components for channel {color_channel}")
        return None

    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    top2 = np.argsort(sizes)[-2:] + 1  # labels are 1-indexed

    lines = []
    for lbl in top2:
        ys, xs = np.where(labeled == lbl)
        pts = np.stack([xs, ys], axis=1).astype(float)
        if len(pts) < 2:
            print(f"  Warning: component too small for channel {color_channel}")
            return None
        lines.append(fit_line_to_points(pts))

    return lines[0], lines[1]


def intersect_lines(l1, l2):
    return np.cross(l1, l2)


def vanishing_point_from_ppm(ppm_path, color_channel):
    result = extract_colored_lines(ppm_path, color_channel)
    if result is None:
        return None
    l1, l2 = result
    return intersect_lines(l1, l2)