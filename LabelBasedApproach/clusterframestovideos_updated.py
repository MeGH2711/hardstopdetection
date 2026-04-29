import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
from numba import njit

@dataclass
class Box:
    cls: str
    poly: np.ndarray  # shape (4, 2)
    aabb: Tuple[float, float, float, float]
    heading: float | None

@dataclass
class FrameData:
    frame_id: str
    label_path: Path
    image_path: Path | None
    flight_height: float | None
    boxes: List[Box]
    frame_aabb: np.ndarray  # [min_x, min_y, max_x, max_y]

# --- NUMBA ACCELERATED GEOMETRY ---

@njit
def polygon_area_jit(poly: np.ndarray) -> float:
    if len(poly) < 3: return 0.0
    x, y = poly[:, 0], poly[:, 1]
    area = 0.0
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        area += x[i] * y[j]
        area -= x[j] * y[i]
    return 0.5 * abs(area)

@njit
def _line_intersection_jit(s, e, a, b):
    # s, e are subject points; a, b are clipper points
    dx, dy = s[0] - e[0], s[1] - e[1]
    dx_clip, dy_clip = a[0] - b[0], a[1] - b[1]
    den = dx * dy_clip - dy * dx_clip
    if abs(den) < 1e-12: 
        return e
    px = ((s[0] * e[1] - s[1] * e[0]) * dx_clip - dx * (a[0] * b[1] - a[1] * b[0])) / den
    py = ((s[0] * e[1] - s[1] * e[0]) * dy_clip - dy * (a[0] * b[1] - a[1] * b[0])) / den
    return np.array([px, py])

@njit
def polygon_clip_jit(subject, clipper):
    # Calculate clipper orientation
    orientation = 0.0
    for i in range(len(clipper)):
        j = (i + 1) % len(clipper)
        orientation += clipper[i, 0] * clipper[j, 1] - clipper[j, 0] * clipper[i, 1]
    orientation_sign = 1.0 if orientation > 0 else -1.0

    output_poly = subject
    for i in range(len(clipper)):
        a = clipper[i]
        b = clipper[(i + 1) % len(clipper)]
        
        input_poly = output_poly
        if len(input_poly) == 0: 
            return np.zeros((0, 2))
            
        # We use a fixed-size array for the intermediate result to satisfy Numba
        # Sutherland-Hodgman can at most double the vertices
        temp_output = np.zeros((len(input_poly) * 2, 2))
        count = 0
        
        s = input_poly[len(input_poly) - 1]
        for j in range(len(input_poly)):
            e = input_poly[j]
            
            # Check "inside" via cross product
            e_inside = ((b[0] - a[0]) * (e[1] - a[1]) - (b[1] - a[1]) * (e[0] - a[0])) * orientation_sign >= -1e-9
            s_inside = ((b[0] - a[0]) * (s[1] - a[1]) - (b[1] - a[1]) * (s[0] - a[0])) * orientation_sign >= -1e-9
            
            if e_inside:
                if not s_inside:
                    temp_output[count] = _line_intersection_jit(s, e, a, b)
                    count += 1
                temp_output[count] = e
                count += 1
            elif s_inside:
                temp_output[count] = _line_intersection_jit(s, e, a, b)
                count += 1
            s = e
        output_poly = temp_output[:count]
        
    return output_poly

@njit
def oriented_iou_jit(poly1, poly2):
    inter = polygon_clip_jit(poly1, poly2)
    if len(inter) < 3: 
        return 0.0
    inter_area = polygon_area_jit(inter)
    a1, a2 = polygon_area_jit(poly1), polygon_area_jit(poly2)
    union = a1 + a2 - inter_area
    return float(inter_area / union) if union > 0 else 0.0

# --- DATA PARSING ---

def parse_label_file(path: Path, images_dir: Path) -> FrameData:
    frame_id = path.stem
    img_path = images_dir / f"{frame_id}.jpg"
    boxes, flight_height = [], None
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except Exception:
        lines = []
    
    if not lines:
        return FrameData(frame_id, path, img_path if img_path.exists() else None, None, [], np.zeros(4))

    data_lines = lines[1:] if lines[0].lower().startswith("flightheight:") else lines
    if lines[0].lower().startswith("flightheight:"):
        try: flight_height = float(lines[0].split(":", 1)[1])
        except: pass

    all_coords = []
    for ln in data_lines:
        parts = ln.split()
        if len(parts) < 9: continue
        try:
            poly = np.array(list(map(float, parts[0:8])), dtype=np.float64).reshape(4, 2)
            aabb = (poly[:,0].min(), poly[:,1].min(), poly[:,0].max(), poly[:,1].max())
            heading = float(parts[10]) if len(parts) >= 11 else None
            boxes.append(Box(cls=parts[8], poly=poly, aabb=aabb, heading=heading))
            all_coords.append(poly)
        except: continue
    
    if all_coords:
        stacked = np.vstack(all_coords)
        frame_aabb = np.array([stacked[:,0].min(), stacked[:,1].min(), 
                               stacked[:,0].max(), stacked[:,1].max()])
    else:
        frame_aabb = np.zeros(4)
        
    return FrameData(frame_id, path, img_path if img_path.exists() else None, flight_height, boxes, frame_aabb)

# --- SIMILARITY CORE ---

def frame_similarity_worker(i, j, f1_boxes, f2_boxes, f1_aabb, f2_aabb, class_aware):
    if (f1_aabb[0] > f2_aabb[2] or f1_aabb[2] < f2_aabb[0] or
        f1_aabb[1] > f2_aabb[3] or f1_aabb[3] < f2_aabb[1]):
        return i, j, 0.0

    def one_way(a_boxes, b_boxes):
        best_scores = []
        for a in a_boxes:
            best = 0.0
            for b in b_boxes:
                if class_aware and a.cls != b.cls: continue
                if (a.aabb[0] > b.aabb[2] or a.aabb[2] < b.aabb[0] or
                    a.aabb[1] > b.aabb[3] or a.aabb[3] < b.aabb[1]): continue
                
                iou = oriented_iou_jit(a.poly, b.poly)
                heading_sim = 1.0
                if a.heading is not None and b.heading is not None:
                    heading_sim = max(0, (np.cos(np.radians(a.heading) - np.radians(b.heading)) + 1) / 2)
                
                score = iou * heading_sim
                if score > best: best = score
            best_scores.append(best)
        return float(np.mean(best_scores)) if best_scores else 0.0

    sim = 0.5 * (one_way(f1_boxes, f2_boxes) + one_way(f2_boxes, f1_boxes))
    return i, j, sim

def frame_similarity_worker_wrapper(args):
    return frame_similarity_worker(*args)

# --- CLUSTERING & OUTPUT ---

def connected_components_from_threshold(sim: np.ndarray, threshold: float) -> List[List[int]]:
    n = sim.shape[0]
    visited, components = [False] * n, []
    for i in range(n):
        if visited[i]: continue
        stack, comp = [i], []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            neighbors = np.where(sim[u] >= threshold)[0]
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        components.append(sorted(comp))
    return components

def sequence_cluster_indices(cluster_indices: List[int], sim: np.ndarray) -> List[int]:
    if len(cluster_indices) <= 1: return cluster_indices[:]
    idx = np.array(cluster_indices, dtype=int)
    sub = sim[np.ix_(idx, idx)]
    start_local = int(np.argmin(sub.sum(axis=1)))
    visited, order = {start_local}, [start_local]
    while len(visited) < len(idx):
        curr = order[-1]
        unvisited = [u for u in range(len(idx)) if u not in visited]
        best_next = max(unvisited, key=lambda u: sub[curr, u])
        if sub[curr, best_next] == 0:
             best_next = max(unvisited, key=lambda u: float(np.max(sub[u, list(visited)])))
        visited.add(best_next)
        order.append(best_next)
    return [int(idx[loc]) for loc in order]

def write_outputs(frames, sim, clusters, output_dir, export_dirs):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c_id, comp in enumerate(tqdm(clusters, desc="Writing results")):
        ordered = sequence_cluster_indices(comp, sim)
        v_id = f"video_{c_id:04d}"
        cluster_path = output_dir / "clustered_frames" / v_id
        if export_dirs: 
            (cluster_path / "images").mkdir(parents=True, exist_ok=True)
            (cluster_path / "labels").mkdir(parents=True, exist_ok=True)
        
        for seq_idx, f_idx in enumerate(ordered):
            fr = frames[f_idx]
            rows.append({"frame_id": fr.frame_id, "video_id": v_id, "sequence_index": seq_idx, "label_path": str(fr.label_path)})
            if export_dirs:
                new_name = f"{seq_idx:06d}_{fr.frame_id}"
                shutil.copy2(fr.label_path, cluster_path / "labels" / f"{new_name}.txt")
                if fr.image_path and fr.image_path.exists():
                    shutil.copy2(fr.image_path, cluster_path / "images" / f"{new_name}.jpg")

    with (output_dir / "frame_video_sequence.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "video_id", "sequence_index", "label_path"])
        writer.writeheader()
        writer.writerows(rows)

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_dir", type=Path, required=True)
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_optimized"))
    parser.add_argument("--sim_threshold", type=float, default=0.22)
    parser.add_argument("--window_size", type=int, default=300)
    parser.add_argument("--class_aware", action="store_true")
    parser.add_argument("--export_cluster_dirs", action="store_true")
    args = parser.parse_args()

    label_files = sorted(args.labels_dir.glob("*.txt"))
    frames = [parse_label_file(p, args.images_dir) for p in tqdm(label_files, desc="Parsing Labels")]
    
    n = len(frames)
    sim_matrix = np.eye(n)
    tasks = [(i, j, frames[i].boxes, frames[j].boxes, frames[i].frame_aabb, frames[j].frame_aabb, args.class_aware) 
             for i in range(n) for j in range(i + 1, min(i + args.window_size, n))]

    print(f"Processing {len(tasks)} pairs...")
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(frame_similarity_worker_wrapper, tasks, chunksize=500), total=len(tasks)))
        for i, j, s in results: 
            sim_matrix[i, j] = sim_matrix[j, i] = s

    clusters = sorted(connected_components_from_threshold(sim_matrix, args.sim_threshold), key=len, reverse=True)
    write_outputs(frames, sim_matrix, clusters, args.output_dir, args.export_cluster_dirs)
    print(f"Done! Clusters found: {len(clusters)}")

if __name__ == "__main__":
    main()