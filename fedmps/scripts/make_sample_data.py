"""Create a tiny synthetic LFW-style folder dataset for testing.
Produces 6 identities × 10 images in a temp-like folder so
01_prepare_data.py can run with --source folder --backbone dummy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / "data" / "sample_faces"
N_IDS = 6
IMGS_PER_ID = 10
SIZE = 96
rng = np.random.default_rng(42)

OUT.mkdir(parents=True, exist_ok=True)

names = [
    "Alice_Smith", "Bob_Jones", "Carol_White",
    "David_Brown", "Eva_Green", "Frank_Black",
]

for pid, name in enumerate(names[:N_IDS]):
    d = OUT / name
    d.mkdir(exist_ok=True)
    # Each identity has a base skin tone; images vary by noise.
    base = rng.integers(80, 200, size=(3,), dtype=np.int16)
    for j in range(IMGS_PER_ID):
        face = np.clip(
            base + rng.normal(0, 15, size=(SIZE, SIZE, 3)),
            0, 255
        ).astype(np.uint8)
        # Add a simple elliptical "head" shape for realism.
        cx, cy = SIZE // 2, SIZE // 2
        cv2.ellipse(face, (cx, cy), (28, 36), 0, 0, 360,
                    (int(base[0]), int(base[1]), int(base[2])), -1)
        face += rng.integers(0, 10, size=face.shape, dtype=np.uint8)
        cv2.imwrite(str(d / f"img_{j:02d}.png"), face)

print(f"Created {N_IDS} identities × {IMGS_PER_ID} images in: {OUT}")
print(f"Now run:")
print(f"  python fedmps/scripts/01_prepare_data.py --backbone dummy --max-identities 6 --source folder --folder \"{OUT}\"")

