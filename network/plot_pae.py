import numpy as np
import torch
import json
from icecream import ic
from matplotlib import pyplot as plt

from density import split_by_pae

pae_array_rf2 = torch.tensor(np.load("test_atpbind_emd_14914_after_dock_cycle_0_split_0.npz")["pae"]).to(torch.float32)

af2_pae_path = "AF-P03960-F1-predicted_aligned_error_v4.json"
pae_array_af2 = torch.tensor(json.load(open(af2_pae_path))[0]["predicted_aligned_error"])

def two_d_cumsum(pae_array: torch.Tensor) -> torch.Tensor:
    pae_cumsum_rows = torch.cumsum(pae_array, dim=0)
    pae_cumsum = torch.cumsum(pae_cumsum_rows, dim=1)
    from torch.nn import functional

    pae_cumsum = functional.pad(input=pae_cumsum, pad=(1, 0, 1, 0), mode='constant', value=0.)
    return pae_cumsum

pae_cumsum_rf2 = two_d_cumsum(pae_array_rf2)
pae_cumsum_af2 = two_d_cumsum(pae_array_af2)


def inter_vs_intra_pae_score(pae_cumsum: torch.Tensor, test_slice: slice) -> float:
    # The letters below represent the cumulative sum of the region as well as the regions to the top and left.
    # | A | B | C |
    # | D | E | F |
    # | G | H | I |

    A = pae_cumsum[test_slice.start, test_slice.start]
    B = pae_cumsum[test_slice.start, test_slice.stop]
    C = pae_cumsum[test_slice.start, -1]
    D = pae_cumsum[test_slice.stop, test_slice.start]
    E = pae_cumsum[test_slice.stop, test_slice.stop]
    F = pae_cumsum[test_slice.stop, -1]
    G = pae_cumsum[-1, test_slice.start]
    H = pae_cumsum[-1, test_slice.stop]

    test_slice_len = test_slice.stop - test_slice.start
    sum_inter_split_pae = A + E - B - D
    sum_intra_split_pae = F + H - C - G - 2 * sum_inter_split_pae

    pae_region_scale_factor = (sum_intra_split_pae * test_slice_len) / (
            (sum_inter_split_pae + 1e-9) * (pae_cumsum.shape[0] - 1 - test_slice_len))
    return pae_region_scale_factor


splits_rf2: list[int] = split_by_pae(pae_array_rf2, min_split_length=100)
splits_af2: list[int] = split_by_pae(pae_array_af2, min_split_length=100)

print(f"{splits_af2=}")
print(f"{splits_rf2=}")

# fig, (ax1, ax2) = plt.subplots(2)
# ax1.imshow(pae_array_af2, cmap="hot", interpolation="nearest")
# ax1.annotate("af2", xy=(0, 1))
# ax2.imshow(pae_array_rf2, cmap="hot", interpolation="nearest")
# ax2.annotate("rf2", xy=(0, 1))
# plt.savefig(fname="pae_plot_af2.png")

print(f"{len(pae_array_rf2)}=")
print(f"{len(pae_array_af2)}=")
print(f"{inter_vs_intra_pae_score(pae_cumsum_rf2, slice(0, 400))}")
print(f"{inter_vs_intra_pae_score(pae_cumsum_af2, slice(0, 400))}")

print(pae_array_rf2.dtype)
