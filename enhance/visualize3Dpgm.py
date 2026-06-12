#!/usr/bin/env python3
"""
polar_to_pointcloud.py

Carica una (o più) immagine PNG polar-grid (W=1024, H=32) in bianco/nero
che codificano la distanza (0..150 m), converte in point cloud 3D e la
visualizza.

Assunzioni prese automaticamente:
- le colonne (W) sono azimut su 360 gradi con passo 360/W
- le righe (H) sono i 32 layer verticali distribuiti linearmente su +/-15° (tot 30°)
  (ossia linspace(-15, +15, H))
- valore pixel 0 -> distanza 0 m; valore pixel 255 -> 150 m, ma i pixel con
  valore esattamente 255 vengono scartati (sono cielo / punti droppati)

Uso:
    python polar_to_pointcloud.py --input polar.png
    python polar_to_pointcloud.py --input folder_with_pngs/ --save-ply output.ply

Dipendenze: numpy, pillow (PIL), matplotlib

"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

MAX_DISTANCE = 150.0  # metri


def load_image_as_array(path: Path):
    img = Image.open(path).convert('L')  # grayscale
    arr = np.array(img, dtype=np.uint8)
    return arr


def polar_to_pointcloud(img_arr: np.ndarray,
                         max_distance: float = MAX_DISTANCE,
                         discard_white: bool = True,
                         h_fov_deg: float = 360.0,
                         v_fov_l_deg: float = -20.0,
                         v_fov_h_deg: float = 20.0,
                         pixel_center: bool = True,
                         axis_order: str = 'xyz'):
    """
    Decode a polar-grid image created with the same projection as your `bin_to_png`.

    This inverts the mapping you provided:
      - pitch = asin(z/d)
      - yaw = get_centered_yaw(x,y)
      - nu = (yaw - h_fov/2) / (-h_fov)
      - nv = 1 - (pitch - v_fov_l) / (v_fov_h - v_fov_l)
      - u = nu * width, v = nv * height
      - pixel intensity -> d = intensity * max_distance / 255

    Parameters added:
      - h_fov_deg, v_fov_l_deg, v_fov_h_deg : match the values used when generating the PNG
      - pixel_center: if True uses u+0.5, v+0.5 to sample pixel centers
      - axis_order: output ordering of coordinates. Default 'xyz'. Other useful value to
        match your original point layout is 'xzy' if your original array used [x,z,y].
    """
    H, W = img_arr.shape

    # intensity -> distance
    # distances = (img_arr.astype(np.float32) / 255.0) * max_distance
    I = img_arr.astype(np.float32)
    distances = max_distance * (1.0 - np.power((255.0 - I) / 255.0, 0.25))

    # prepare u,v normalized coordinates
    if pixel_center:
        u_idx = (np.arange(W) + 0.5) / W  # nu = u/width
        v_idx = (np.arange(H) + 0.5) / H  # nv = v/height
    else:
        u_idx = (np.arange(W)) / W
        v_idx = (np.arange(H)) / H

    # invert nu->yaw: yaw = h_fov*(0.5 - nu)
    h_fov = np.deg2rad(h_fov_deg)
    yaw = h_fov * (0.5 - u_idx)  # shape (W,)

    # invert nv->pitch: pitch = v_fov_l + (1 - nv) * (v_fov_h - v_fov_l)
    v_fov_l = np.deg2rad(v_fov_l_deg)
    v_fov_h = np.deg2rad(v_fov_h_deg)
    pitch = v_fov_l + (1.0 - v_idx) * (v_fov_h - v_fov_l)  # shape (H,)

    # meshgrid to HxW
    yaw_grid = np.broadcast_to(yaw[np.newaxis, :], (H, W))
    pitch_grid = np.broadcast_to(pitch[:, np.newaxis], (H, W))

    # masks: discard white (255) and zero distances
    mask = np.ones_like(distances, dtype=bool)
    if discard_white:
        mask &= (img_arr != 255)
    mask &= (distances > 0.0)

    r = distances[mask]
    yaw_m = yaw_grid[mask]
    pitch_m = pitch_grid[mask]

    # reconstruct cartesian coordinates used by the original bin_to_png:
    # z = d * sin(pitch)
    # r_xy = d * cos(pitch)
    # x = r_xy * cos(yaw)
    # y = r_xy * sin(yaw)
    z = r * np.sin(pitch_m)
    r_xy = r * np.cos(pitch_m)
    x = r_xy * np.cos(yaw_m)
    y = r_xy * np.sin(yaw_m)

    # arrange axis order to match user's original structure if needed
    if axis_order == 'xyz':
        pts = np.stack([x, y, z], axis=1).astype(np.float32)
    elif axis_order == 'xzy':
        pts = np.stack([x, z, y], axis=1).astype(np.float32)
    else:
        # generic reorder if user passes e.g. 'yxz' etc
        arr = {'x': x, 'y': y, 'z': z}
        pts = np.stack([arr[a] for a in axis_order], axis=1).astype(np.float32)

    return pts, r


def save_ply_ascii(path: Path, points: np.ndarray, intensities: np.ndarray = None):
    """Salva point cloud in formato PLY ASCII (vertex only).
    Optionally memorizza intensities come property float "intensity".
    """
    n = points.shape[0]
    with open(path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {n}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if intensities is not None:
            f.write('property float intensity\n')
        f.write('end_header\n')
        if intensities is None:
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        else:
            for p, it in zip(points, intensities):
                f.write(f"{p[0]} {p[1]} {p[2]} {it}\n")


def visualize_pointcloud(points: np.ndarray, intensities: np.ndarray = None, point_size: float = 1.0):
    fig = plt.figure(figsize=(12, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    # distanza dal centro per color mapping
    distances = np.linalg.norm(points, axis=1)

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=distances if intensities is None else intensities,
        s=point_size,
        cmap='inferno',
        depthshade=False
    )

    # ===== CAMERA ORIENTATION (HARD-CODED) =====
    #ax.view_init(elev=32, azim=-200, roll=0)
    ax.view_init(elev=90, azim=-200, roll=0)

    # ===== RIMOZIONE ASSI =====
    ax.set_axis_off()

    # ===== ASPETTO A TUTTO SCHERMO =====
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # ===== COMPUTE BASE BOUNDS =====
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() * 0.5

    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    # =====================================================================
    # HARD-CODED ZOOM & PAN (PUNTO DI INTERESSE)
    # =====================================================================
    # Modifica questi valori per scegliere DOVE guardare e QUANTO zoomare

    ZOOM_FACTOR = 0.5 # ZOOM_FACTOR = 0.28        # < 1 = più zoom, > 1 = più lontano
    PAN_X = -20.0              # spostamento in metri sull'asse X
    PAN_Y = -15.0              # spostamento in metri sull'asse Y
    PAN_Z = -12.0              # spostamento in metri sull'asse Z

    final_range = max_range * ZOOM_FACTOR

    cx = mid_x + PAN_X
    cy = mid_y + PAN_Y
    cz = mid_z + PAN_Z

    ax.set_xlim(cx - final_range, cx + final_range)
    ax.set_ylim(cy - final_range, cy + final_range)
    ax.set_zlim(cz - final_range, cz + final_range)

    plt.show()


def find_pngs(input_path: Path):
    if input_path.is_dir():
        return sorted([p for p in input_path.glob('*.png')])
    else:
        return [input_path]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='./sequences/from_my_simulator/', help='PNG file or folder with PNGs')
    p.add_argument('--save-ply', '-o', default=None, help='Optional: salva point cloud in PLY ASCII')
    p.add_argument('--point-size', type=float, default=1.0, help='dimensione punti per la visualizzazione')
    p.add_argument('--vertical-fov', type=float, default=40.0, help='FoV verticale totale in gradi')
    p.add_argument('--vertical-center', type=float, default=0.0, help='centro FoV verticale in gradi')
    args = p.parse_args()

    input_path = Path(args.input)
    imgs = find_pngs(input_path)
    if len(imgs) == 0:
        raise SystemExit(f'Non ho trovato PNG in: {input_path}')

    all_points = []
    all_int = []
    for img_path in imgs:
        arr = load_image_as_array(img_path)
        pts, ints = polar_to_pointcloud(arr,
                                       max_distance=MAX_DISTANCE,
                                       discard_white=True,
                                       v_fov_h_deg = args.vertical_center + args.vertical_fov/2.0,
                                       v_fov_l_deg = args.vertical_center - args.vertical_fov/2.0)
        all_points.append(pts)
        all_int.append(ints)

    points = np.vstack(all_points)
    intensities = np.hstack(all_int)

    print(f'Loaded {len(imgs)} images, total points: {points.shape[0]}')

    if args.save_ply:
        save_ply_ascii(Path(args.save_ply), points, intensities)
        print(f'Saved PLY to {args.save_ply}')

    visualize_pointcloud(points, intensities, point_size=args.point_size)


if __name__ == '__main__':
    main()
