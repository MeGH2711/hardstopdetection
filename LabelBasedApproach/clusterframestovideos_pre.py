#!/usr/bin/env python3
"""
Cluster shuffled frame-level annotations into video groups using max-IoU similarity.
Updated with tqdm real-time progress tracking.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm  # Added for progress tracking


@dataclass
class Box:
    cls: str
    poly: np.ndarray  # shape (4, 2)


@dataclass
class FrameData:
    frame_id: str
    label_path: Path
    image_path: Path | None
    flight_height: float | None
    boxes: List[Box]


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _inside(p: np.ndarray, a: np.ndarray, b: np.ndarray, orientation_sign: float) -> bool:
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return cross * orientation_sign >= -1e-9


def _line_intersection(s: np.ndarray, e: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1 = s
    x2, y2 = e
    x3, y3 = a
    x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return e
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return np.array([px, py], dtype=np.float64)


def polygon_clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    output = subject.copy()
    if len(output) == 0: return output
    orientation = np.sum(clipper[:, 0] * np.roll(clipper[:, 1], -1) - clipper[:, 1] * np.roll(clipper[:, 0], -1))
    orientation_sign = 1.0 if orientation > 0 else -1.0
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        input_list, output = output, []
        if len(input_list) == 0: break
        s = input_list[-1]
        for e in input_list:
            if _inside(e, a, b, orientation_sign):
                if not _inside(s, a, b, orientation_sign):
                    output.append(_line_intersection(s, e, a, b))
                output.append(e)
            elif _inside(s, a, b, orientation_sign):
                output.append(_line_intersection(s, e, a, b))
            s = e
        output = np.array(output, dtype=np.float64) if output else np.empty((0, 2), dtype=np.float64)
    return output


def oriented_iou(poly1: np.ndarray, poly2: np.ndarray) -> float:
    inter = polygon_clip(poly1, poly2)
    if len(inter) < 3: return 0.0
    inter_area = polygon_area(inter)
    a1, a2 = polygon_area(poly1), polygon_area(poly2)
    union = a1 + a2 - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def parse_label_file(path: Path, images_dir: Path) -> FrameData:
    frame_id = path.stem
    img_path = images_dir / f"{frame_id}.jpg"
    if not img_path.exists(): img_path = None
    boxes, flight_height = [], None
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines: return FrameData(frame_id, path, img_path, flight_height, boxes)
    if lines[0].lower().startswith("flightheight:"):
        try: flight_height = float(lines[0].split(":", 1)[1])
        except ValueError: pass
        data_lines = lines[1:]
    else: data_lines = lines
    for ln in data_lines:
        parts = ln.split()
        if len(parts) < 11: continue
        try:
            poly = np.array(list(map(float, parts[0:8])), dtype=np.float64).reshape(4, 2)
            boxes.append(Box(cls=parts[8], poly=poly))
        except ValueError: continue
    return FrameData(frame_id, path, img_path, flight_height, boxes)


def frame_similarity_max_iou(f1: FrameData, f2: FrameData, class_aware: bool = True) -> float:
    if not f1.boxes or not f2.boxes: return 0.0
    def one_way(a_boxes: List[Box], b_boxes: List[Box]) -> float:
        best_scores = []
        for a in a_boxes:
            best = 0.0
            for b in b_boxes:
                if class_aware and a.cls != b.cls: continue
                iou = oriented_iou(a.poly, b.poly)
                if iou > best: best = iou
            best_scores.append(best)
        return float(np.mean(best_scores)) if best_scores else 0.0
    return 0.5 * (one_way(f1.boxes, f2.boxes) + one_way(f2.boxes, f1.boxes))


def build_similarity_matrix(frames: List[FrameData], class_aware: bool = True) -> np.ndarray:
    n = len(frames)
    sim = np.eye(n, dtype=np.float64)
    # Added tqdm to the outer loop of the O(n^2) operation
    for i in tqdm(range(n), desc="Calculating similarities"):
        for j in range(i + 1, n):
            s = frame_similarity_max_iou(frames[i], frames[j], class_aware=class_aware)
            sim[i, j] = sim[j, i] = s
    return sim


def connected_components_from_threshold(sim: np.ndarray, threshold: float) -> List[List[int]]:
    n = sim.shape[0]
    visited = [False] * n
    components = []
    # Using a progress bar based on frames visited
    pbar = tqdm(total=n, desc="Clustering frames")
    for i in range(n):
        if visited[i]: continue
        stack, comp = [i], []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            pbar.update(1)
            neighbors = np.where(sim[u] >= threshold)[0]
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        components.append(sorted(comp))
    pbar.close()
    return components


def sequence_cluster_indices(cluster_indices: List[int], sim: np.ndarray) -> List[int]:
    if len(cluster_indices) <= 1: return cluster_indices[:]
    idx = np.array(cluster_indices, dtype=int)
    sub = sim[np.ix_(idx, idx)]
    start_local = int(np.argmin(sub.sum(axis=1)))
    visited, order_local = {start_local}, [start_local]
    while len(visited) < len(idx):
        current = order_local[-1]
        unvisited = [u for u in range(len(idx)) if u not in visited]
        best_next = max(unvisited, key=lambda u: sub[current, u])
        chosen = best_next if sub[current, best_next] > 0 else max(unvisited, key=lambda u: float(np.max(sub[u, list(visited)])))
        visited.add(chosen)
        order_local.append(chosen)
    return [int(idx[loc]) for loc in order_local]


def write_outputs(frames: List[FrameData], sim: np.ndarray, clusters: List[List[int]], output_dir: Path, export_cluster_dirs: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, summary_rows = [], []

    # Added tqdm for cluster output generation
    for c_id, comp in enumerate(tqdm(clusters, desc="Exporting clusters")):
        ordered = sequence_cluster_indices(comp, sim)
        density = float(np.mean(sim[np.ix_(comp, comp)])) if comp else 0.0
        video_id = f"video_{c_id:04d}"
        summary_rows.append({"video_id": video_id, "num_frames": len(comp), "avg_internal_similarity": round(density, 6)})
        
        cluster_path = output_dir / "clustered_frames" / video_id
        if export_cluster_dirs:
            (cluster_path / "images").mkdir(parents=True, exist_ok=True)
            (cluster_path / "labels").mkdir(parents=True, exist_ok=True)

        for seq_idx, frame_idx in enumerate(ordered):
            fr = frames[frame_idx]
            rows.append({
                "frame_id": fr.frame_id, "video_id": video_id, "sequence_index": seq_idx,
                "label_path": str(fr.label_path), "image_path": str(fr.image_path) or "",
                "num_boxes": len(fr.boxes), "flight_height": fr.flight_height if fr.flight_height is not None else ""
            })
            if export_cluster_dirs:
                new_name = f"{seq_idx:06d}_{fr.frame_id}"
                if fr.image_path and fr.image_path.exists():
                    shutil.copy2(fr.image_path, cluster_path / "images" / f"{new_name}.jpg")
                shutil.copy2(fr.label_path, cluster_path / "labels" / f"{new_name}.txt")

    with (output_dir / "frame_video_sequence.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "video_id", "sequence_index", "label_path", "image_path", "num_boxes", "flight_height"])
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "clusters_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "num_frames", "avg_internal_similarity"])
        writer.writeheader()
        writer.writerows(summary_rows)


def load_frames(labels_dir: Path, images_dir: Path, max_frames: int | None = None) -> List[FrameData]:
    label_files = sorted(labels_dir.glob("*.txt"))
    if max_frames is not None: label_files = label_files[:max_frames]
    # Added tqdm for file parsing
    return [parse_label_file(p, images_dir) for p in tqdm(label_files, desc="Parsing label files")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster shuffled frame annotations into videos using max-IoU.")
    parser.add_argument("--labels_dir", type=Path, required=True)
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_iou_clusters"))
    parser.add_argument("--sim_threshold", type=float, default=0.22)
    parser.add_argument("--class_aware", action="store_true")
    parser.add_argument("--export_cluster_dirs", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    if not args.labels_dir.exists() or not args.images_dir.exists():
        raise FileNotFoundError("Check labels/images directory paths.")

    frames = load_frames(args.labels_dir, args.images_dir, max_frames=args.max_frames)
    if not frames: raise RuntimeError("No label files found.")

    sim = build_similarity_matrix(frames, class_aware=args.class_aware)
    clusters = connected_components_from_threshold(sim, threshold=args.sim_threshold)
    clusters.sort(key=len, reverse=True)

    write_outputs(frames, sim, clusters, args.output_dir, args.export_cluster_dirs)
    print(f"\nDone. Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()