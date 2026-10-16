from typing import List, Tuple


def build_vertical_stack(base_gx: int, base_gy: int, base_gz: int, height_voxels: int) -> List[Tuple[int, int, int]]:
    voxels = []
    for i in range(height_voxels):
        voxels.append((base_gx, base_gy, base_gz + i))
    return voxels