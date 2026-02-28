"""
Preprocessing Script for JetClass Dataset
==========================================
Handles the filename-based class structure where each ROOT file
belongs to exactly one class (e.g. HToBB_120.root → class Hbb).

Confirmed branch names (from ZJetsToNuNu_120.root inspection):
---------------------------------------------------------------
  part_px, part_py   →  pt = sqrt(px²+py²)
  part_deta          →  eta relative to jet axis
  part_dphi          →  phi relative to jet axis
  part_energy        →  particle energy

Stored particle feature layout: [pt, deta, dphi, energy]
  This is the standard JetClass input representation used in the paper.

File → Class mapping
--------------------
  ZJetsToNuNu  → 0  QCD
  HToBB        → 1  Hbb
  HToCC        → 2  Hcc
  HToGG        → 3  Hgg
  HToWW4Q      → 4  H4q
  HToWW2Q1L    → 5  Hqql
  ZToQQ        → 6  Zqq
  WToQQ        → 7  Wqq
  TTBarLep     → 8  Tbl   (matched before TTBar — longest prefix wins)
  TTBar        → 9  Tbq

Pipeline
--------
  1. Scan input directory, group files by class prefix
  2. Auto-detect TTree name and branch names from first file
  3. Load each class independently (cap = num_events // n_classes)
  4. Sanity-check NaN/Inf
  5. Vectorised invariant-mass computation
  6. Balanced class sampling  (subsample to smallest class)
  7. Per-class independent 80/10/10 split → shuffle → save .npz + metadata

Usage
-----
  python scripts/preprocess_data.py \\
      --input-dir  ./data/raw \\
      --output-dir ./data/jetclass_100k \\
      --num-events 100000 \\
      --max-particles 128 \\
      --train-split 0.8 \\
      --val-split   0.1 \\
      --test-split  0.1 \\
      --seed 42

Author: GSoC 2026 Candidate
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

try:
    import uproot
    import awkward as ak
    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False
    print("Warning: uproot not installed. Run:  pip install uproot awkward")


# ---------------------------------------------------------------------------
# Class map
# ORDER MATTERS: TTBarLep must appear before TTBar (longest prefix wins).
# ---------------------------------------------------------------------------
CLASS_MAP: List[Tuple[str, int, str]] = [
    # (filename_prefix,  class_id,  short_name)
    ("ZJetsToNuNu",  0, "QCD"),
    ("HToBB",        1, "Hbb"),
    ("HToCC",        2, "Hcc"),
    ("HToGG",        3, "Hgg"),
    ("HToWW4Q",      4, "H4q"),
    ("HToWW2Q1L",    5, "Hqql"),
    ("ZToQQ",        6, "Zqq"),
    ("WToQQ",        7, "Wqq"),
    ("TTBarLep",     8, "Tbl"),   # <-- before TTBar
    ("TTBar",        9, "Tbq"),
]

CLASS_MAP_SORTED = sorted(CLASS_MAP, key=lambda x: len(x[0]), reverse=True)
ID_TO_NAME = {cid: name for _, cid, name in CLASS_MAP}
N_CLASSES  = len(CLASS_MAP)


# ---------------------------------------------------------------------------
# Filename → class
# ---------------------------------------------------------------------------

def infer_class_from_filename(filename: str) -> Tuple[int, str]:
    """
    Match the file stem against CLASS_MAP_SORTED (longest prefix first).
    Returns (class_id, class_name) or raises ValueError.
    """
    stem = Path(filename).stem
    for prefix, cid, name in CLASS_MAP_SORTED:
        if stem.startswith(prefix):
            return cid, name
    raise ValueError(
        f"Cannot infer class from '{filename}'.\n"
        f"Known prefixes: {[p for p, _, _ in CLASS_MAP]}"
    )


def group_files_by_class(input_dir: Path) -> Dict[int, List[Path]]:
    """Return {class_id: [Path, ...]} for all *.root files in input_dir."""
    root_files = sorted(input_dir.glob("*.root"))
    if not root_files:
        raise FileNotFoundError(f"No .root files found in {input_dir}")

    groups: Dict[int, List[Path]] = defaultdict(list)
    skipped = []

    for f in root_files:
        try:
            cid, _ = infer_class_from_filename(f.name)
            groups[cid].append(f)
        except ValueError:
            skipped.append(f.name)

    if skipped:
        print(f"  ⚠  Skipping {len(skipped)} unrecognised file(s): {skipped}")

    print(f"\n  Files grouped by class:")
    for cid in sorted(groups):
        print(f"    [{cid}] {ID_TO_NAME[cid]:>6s} : {len(groups[cid])} file(s)")

    return groups


# ---------------------------------------------------------------------------
# Branch auto-detection
# ---------------------------------------------------------------------------

_TREE_CANDIDATES  = ["tree", "jets", "JetTree", "Events"]

# pt is DERIVED: sqrt(part_px**2 + part_py**2)
_PX_CANDIDATES    = ["part_px",   "Particle.Px"]
_PY_CANDIDATES    = ["part_py",   "Particle.Py"]
# eta/phi stored relative to jet axis
_DETA_CANDIDATES  = ["part_deta", "part_eta",   "Particle.Eta"]
_DPHI_CANDIDATES  = ["part_dphi", "part_phi",   "Particle.Phi"]
_ENERGY_CANDIDATES= ["part_energy","part_e",    "Particle.E", "Particle.Energy", "part_E"]


def detect_branches(root_file: Path) -> Tuple[str, str, str, str, str, str]:
    """
    Open one ROOT file and return:
        (tree_name, px_b, py_b, deta_b, dphi_b, e_b)

    pt is computed downstream as sqrt(px²+py²).
    Prints detected names so users can verify.
    """
    with uproot.open(root_file) as f:

        # Locate TTree
        tree_name = None
        for cand in _TREE_CANDIDATES:
            if cand in f:
                tree_name = cand
                break
        if tree_name is None:
            for key in f.keys():
                try:
                    if hasattr(f[key], "keys"):
                        tree_name = key
                        break
                except Exception:
                    pass
        if tree_name is None:
            raise RuntimeError(
                f"No TTree found in {root_file}.\nKeys: {list(f.keys())}"
            )

        present = set(f[tree_name].keys())

        def _pick(candidates, label):
            for c in candidates:
                if c in present:
                    return c
            raise RuntimeError(
                f"No {label} branch found in {root_file}.\n"
                f"Tried: {candidates}\n"
                f"Available (first 50): {sorted(present)[:50]}"
            )

        px_b   = _pick(_PX_CANDIDATES,    "px")
        py_b   = _pick(_PY_CANDIDATES,    "py")
        deta_b = _pick(_DETA_CANDIDATES,  "deta")
        dphi_b = _pick(_DPHI_CANDIDATES,  "dphi")
        e_b    = _pick(_ENERGY_CANDIDATES,"energy")

    print(f"\n  Auto-detected from '{root_file.name}':")
    print(f"    TTree  : {tree_name}")
    print(f"    px     : {px_b}   ─┐")
    print(f"    py     : {py_b}   ─┴─ pt = sqrt(px²+py²)")
    print(f"    deta   : {deta_b}  (relative to jet axis)")
    print(f"    dphi   : {dphi_b}  (relative to jet axis)")
    print(f"    energy : {e_b}")

    return tree_name, px_b, py_b, deta_b, dphi_b, e_b


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def compute_jet_masses(
    particles:     np.ndarray,  # (N, max_p, 4)  cols = [pt, deta, dphi, E]
    num_particles: np.ndarray,  # (N,)
) -> np.ndarray:
    """
    Vectorised invariant mass.
    Uses stored [pt, deta, dphi, E] where deta/dphi are relative to jet axis.

    m² = (ΣE)² – (Σpx)² – (Σpy)² – (Σpz)²
    """
    pT   = particles[:, :, 0]
    deta = particles[:, :, 1]
    dphi = particles[:, :, 2]
    E    = particles[:, :, 3]

    # Mask padded slots
    idx  = np.arange(particles.shape[1])[None, :]
    mask = (idx < num_particles[:, None]).astype(np.float32)

    px = pT * np.cos(dphi)  * mask
    py = pT * np.sin(dphi)  * mask
    pz = pT * np.sinh(deta) * mask
    E  = E  * mask

    m2 = E.sum(1)**2 - px.sum(1)**2 - py.sum(1)**2 - pz.sum(1)**2
    return np.sqrt(np.maximum(0.0, m2)).astype(np.float32)


def sanity_check(particles: np.ndarray, labels: np.ndarray):
    """Zero out NaN/Inf values and print class distribution."""
    bad = ~np.isfinite(particles)
    if bad.any():
        n_bad = bad.any(axis=(1, 2)).sum()
        print(f"  ⚠  {n_bad} jets contained NaN/Inf — zeroed out")
        particles[bad] = 0.0

    print(f"\n  Class distribution after loading:")
    for cid in range(N_CLASSES):
        n = (labels == cid).sum()
        if n > 0:
            print(f"    [{cid}] {ID_TO_NAME[cid]:>6s} : {n:>8,}")


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class JetClassPreprocessor:
    """
    End-to-end preprocessor for the filename-separated JetClass dataset.

    Parameters
    ----------
    input_dir     : directory containing *.root files
    output_dir    : where to write train.npz / val.npz / test.npz
    max_particles : truncate/pad each jet to this many particles (default 128)
    num_events    : total event cap split evenly across classes (default 100k)
    seed          : random seed (default 42)
    """

    def __init__(
        self,
        input_dir:     Path,
        output_dir:    Path,
        max_particles: int = 128,
        num_events:    int = 100_000,
        seed:          int = 42,
    ):
        self.input_dir     = Path(input_dir)
        self.output_dir    = Path(output_dir)
        self.max_particles = max_particles
        self.num_events    = num_events
        self.seed          = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_one_class(
        self,
        files:      List[Path],
        class_id:   int,
        max_events: int,
        tree_name:  str,
        px_b: str, py_b: str, deta_b: str, dphi_b: str, e_b: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load up to max_events jets from same-class ROOT files.

        Stored layout: particles[i, :, :] = [pt, deta, dphi, energy]
          pt   = sqrt(px² + py²)
          deta = part_deta  (relative to jet axis)
          dphi = part_dphi  (relative to jet axis)

        Returns
        -------
        particles     : (n, max_particles, 4)  float32
        labels        : (n,)                   int32
        num_particles : (n,)                   int32
        """
        out_p  = np.zeros((max_events, self.max_particles, 4), dtype=np.float32)
        out_n  = np.zeros(max_events, dtype=np.int32)
        loaded = 0

        for fpath in tqdm(files, desc=f"  [{class_id}] {ID_TO_NAME[class_id]}", leave=False):
            if loaded >= max_events:
                break
            remaining = max_events - loaded

            with uproot.open(fpath) as f:
                tree      = f[tree_name]
                n_entries = min(tree.num_entries, remaining)

                arrays = tree.arrays(
                    [px_b, py_b, deta_b, dphi_b, e_b],
                    entry_stop=n_entries,
                    library="ak",
                )

                for i in range(n_entries):
                    px_i  = ak.to_numpy(arrays[px_b][i]).astype(np.float32)
                    py_i  = ak.to_numpy(arrays[py_b][i]).astype(np.float32)
                    n_p   = min(len(px_i), self.max_particles)
                    slot  = loaded + i

                    out_n[slot]          = n_p
                    out_p[slot, :n_p, 0] = np.sqrt(px_i[:n_p]**2 + py_i[:n_p]**2)  # pt
                    out_p[slot, :n_p, 1] = ak.to_numpy(arrays[deta_b][i])[:n_p].astype(np.float32)
                    out_p[slot, :n_p, 2] = ak.to_numpy(arrays[dphi_b][i])[:n_p].astype(np.float32)
                    out_p[slot, :n_p, 3] = ak.to_numpy(arrays[e_b][i])[:n_p].astype(np.float32)

                loaded += n_entries

        out_p = out_p[:loaded]
        out_n = out_n[:loaded]
        out_l = np.full(loaded, class_id, dtype=np.int32)
        return out_p, out_l, out_n

    def load_all_classes(
        self,
        groups:    Dict[int, List[Path]],
        tree_name: str,
        px_b: str, py_b: str, deta_b: str, dphi_b: str, e_b: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load every class. Per-class cap = num_events // N_CLASSES.
        """
        per_class = max(1, self.num_events // N_CLASSES)
        print(f"\n  Per-class loading cap : {per_class:,} jets\n")

        all_p, all_l, all_n = [], [], []

        for cid in sorted(groups):
            p, l, n = self._load_one_class(
                groups[cid], cid, per_class,
                tree_name, px_b, py_b, deta_b, dphi_b, e_b,
            )
            all_p.append(p)
            all_l.append(l)
            all_n.append(n)
            print(f"  [{cid}] {ID_TO_NAME[cid]:>6s}  →  {len(l):,} jets loaded")

        particles     = np.concatenate(all_p, axis=0)
        labels        = np.concatenate(all_l, axis=0)
        num_particles = np.concatenate(all_n, axis=0)

        print(f"\n  ✓ Total jets loaded : {len(labels):,}")
        return particles, labels, num_particles

    # ------------------------------------------------------------------
    # Balanced splits
    # ------------------------------------------------------------------

    def create_balanced_splits(
        self,
        data:       Dict[str, np.ndarray],
        train_frac: float = 0.8,
        val_frac:   float = 0.1,
        test_frac:  float = 0.1,
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Subsample every class to min_count, split each independently,
        then concatenate and shuffle each split.

        Using exact remainders avoids rounding drift across splits.
        """
        rng    = np.random.default_rng(self.seed)
        labels = data["labels"]

        present  = np.unique(labels)
        counts   = {c: int((labels == c).sum()) for c in present}
        min_cnt  = min(counts.values())

        print(f"\n  Class counts before balancing:")
        for c in present:
            flag = "  ← min" if counts[c] == min_cnt else ""
            print(f"    [{c}] {ID_TO_NAME[c]:>6s} : {counts[c]:>8,}{flag}")

        n_train = int(min_cnt * train_frac)
        n_val   = int(min_cnt * val_frac)
        n_test  = min_cnt - n_train - n_val   # exact remainder, no drift

        print(f"\n  Subsampling to {min_cnt:,} per class")
        print(f"  → {n_train} train  /  {n_val} val  /  {n_test} test  per class")

        tr_idx, va_idx, te_idx = [], [], []

        for c in present:
            idx    = np.where(labels == c)[0]
            chosen = rng.choice(idx, size=min_cnt, replace=False)
            chosen = rng.permutation(chosen)

            tr_idx.append(chosen[:n_train])
            va_idx.append(chosen[n_train : n_train + n_val])
            te_idx.append(chosen[n_train + n_val :])

        # Shuffle across classes
        tr_idx = rng.permutation(np.concatenate(tr_idx))
        va_idx = rng.permutation(np.concatenate(va_idx))
        te_idx = rng.permutation(np.concatenate(te_idx))

        def _sel(idx):
            return {k: v[idx] for k, v in data.items()}

        n_cls = len(present)
        print(f"\n  ✓ Final balanced splits:")
        print(f"    Train : {len(tr_idx):>8,}  ({n_train} × {n_cls} classes)")
        print(f"    Val   : {len(va_idx):>8,}  ({n_val}   × {n_cls} classes)")
        print(f"    Test  : {len(te_idx):>8,}  ({n_test}  × {n_cls} classes)")

        return _sel(tr_idx), _sel(va_idx), _sel(te_idx)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save_split(self, data: Dict[str, np.ndarray], name: str):
        path = self.output_dir / f"{name}.npz"
        np.savez_compressed(path, **data)
        mb = path.stat().st_size / 1e6
        print(f"  ✓ {name:>5s}.npz  →  {path}  ({mb:.1f} MB)")

    def save_metadata(
        self,
        train: Dict, val: Dict, test: Dict,
        train_frac: float, val_frac: float, test_frac: float,
    ):
        lines = [
            "# JetClass preprocessing metadata",
            f"seed:           {self.seed}",
            f"max_particles:  {self.max_particles}",
            f"train_frac:     {train_frac}",
            f"val_frac:       {val_frac}",
            f"test_frac:      {test_frac}",
            f"n_classes:      {N_CLASSES}",
            f"feature_layout: [pt, deta, dphi, energy]",
            f"class_names:    {[ID_TO_NAME[i] for i in range(N_CLASSES)]}",
            f"train_samples:  {len(train['labels'])}",
            f"val_samples:    {len(val['labels'])}",
            f"test_samples:   {len(test['labels'])}",
            "",
            "# Samples per class in train split (should be equal):",
        ]
        for cid in range(N_CLASSES):
            n = int((train["labels"] == cid).sum())
            lines.append(f"  [{cid}] {ID_TO_NAME[cid]:>6s}: {n}")

        path = self.output_dir / "metadata.txt"
        path.write_text("\n".join(lines))
        print(f"  ✓ metadata   →  {path}")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def preprocess(
        self,
        train_frac: float = 0.8,
        val_frac:   float = 0.1,
        test_frac:  float = 0.1,
    ):
        if not HAS_UPROOT:
            raise ImportError("Run:  pip install uproot awkward")

        print("=" * 70)
        print("JetClass Preprocessing  –  filename labels  –  balanced classes")
        print("=" * 70)

        # 1 ── Group files by class
        print("\n[1/6]  Grouping files by class …")
        groups = group_files_by_class(self.input_dir)

        missing = [ID_TO_NAME[c] for c in range(N_CLASSES) if c not in groups]
        if missing:
            print(f"  ⚠  No files found for classes: {missing}")

        # 2 ── Auto-detect branches from first available file
        print("\n[2/6]  Detecting ROOT branches …")
        first_file = groups[sorted(groups)[0]][0]
        tree_name, px_b, py_b, deta_b, dphi_b, e_b = detect_branches(first_file)

        # 3 ── Load all classes
        print("\n[3/6]  Loading jets …")
        particles, labels, num_particles = self.load_all_classes(
            groups, tree_name, px_b, py_b, deta_b, dphi_b, e_b
        )

        # 4 ── Sanity check
        print("\n[4/6]  Sanity checking …")
        sanity_check(particles, labels)

        # 5 ── Compute jet masses (vectorised)
        print("\n[5/6]  Computing jet masses …")
        jet_masses = compute_jet_masses(particles, num_particles)
        print(f"  mean = {jet_masses.mean():.2f} GeV  "
              f"std = {jet_masses.std():.2f} GeV  "
              f"max = {jet_masses.max():.2f} GeV")

        data = {
            "particles":     particles,       # (N, max_p, 4)  float32  [pt,deta,dphi,E]
            "labels":        labels,           # (N,)           int32
            "num_particles": num_particles,    # (N,)           int32
            "jet_masses":    jet_masses,       # (N,)           float32
        }

        # 6 ── Balanced splits
        print("\n[6/6]  Creating balanced splits …")
        train_data, val_data, test_data = self.create_balanced_splits(
            data, train_frac, val_frac, test_frac
        )

        # 7 ── Save
        print("\nSaving outputs …")
        self.save_split(train_data, "train")
        self.save_split(val_data,   "val")
        self.save_split(test_data,  "test")
        self.save_metadata(train_data, val_data, test_data,
                           train_frac, val_frac, test_frac)

        print("\n" + "=" * 70)
        print("✓  All done!")
        print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess JetClass dataset (filename labels, balanced splits)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir",     required=True,               help="Directory with *.root files")
    parser.add_argument("--output-dir",    required=True,               help="Where to write .npz outputs")
    parser.add_argument("--num-events",    type=int,   default=100_000,  help="Total events (split evenly per class)")
    parser.add_argument("--max-particles", type=int,   default=128,      help="Max particles per jet (zero-padded)")
    parser.add_argument("--train-split",   type=float, default=0.8,      help="Training fraction")
    parser.add_argument("--val-split",     type=float, default=0.1,      help="Validation fraction")
    parser.add_argument("--test-split",    type=float, default=0.1,      help="Test fraction")
    parser.add_argument("--seed",          type=int,   default=42,       help="Random seed")

    args = parser.parse_args()

    total = args.train_split + args.val_split + args.test_split
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Splits must sum to 1.0, got {total:.6f}")

    JetClassPreprocessor(
        input_dir     = args.input_dir,
        output_dir    = args.output_dir,
        max_particles = args.max_particles,
        num_events    = args.num_events,
        seed          = args.seed,
    ).preprocess(
        train_frac = args.train_split,
        val_frac   = args.val_split,
        test_frac  = args.test_split,
    )


if __name__ == "__main__":
    main()