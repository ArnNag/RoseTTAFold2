import numpy as np
import torch
import json
from icecream import ic
from matplotlib import pyplot as plt

from density import split_by_pae, compute_pae_enrichment

# pae_array_rf2 = torch.tensor(np.load("test_atpbind_emd_14914_after_dock_cycle_0_split_0.npz")["pae"]).to(torch.float32)

af2_pae_path = "/home/nagleam/Downloads/AF-T0D3N5-F1-predicted_aligned_error_v4.json"
pae_array_af2 = torch.tensor(json.load(open(af2_pae_path))[0]["predicted_aligned_error"], dtype=torch.float)

def two_d_cumsum(pae_array: torch.Tensor) -> torch.Tensor:
    pae_cumsum_rows = torch.cumsum(pae_array, dim=0)
    pae_cumsum = torch.cumsum(pae_cumsum_rows, dim=1)
    from torch.nn import functional

    pae_cumsum = functional.pad(input=pae_cumsum, pad=(1, 0, 1, 0), mode='constant', value=0.)
    return pae_cumsum

# pae_cumsum_rf2 = two_d_cumsum(pae_array_rf2)
pae_cumsum_af2 = two_d_cumsum(pae_array_af2)


def inter_vs_intra_pae_score_naive(pae_array: torch.Tensor, test_slice: slice) -> float:

    assert test_slice.start >= 0
    assert test_slice.stop <= pae_array.shape[0]

    avg_inter_split_pae = torch.mean(pae_array[test_slice, test_slice])
    first_cross_term = pae_array[test_slice, 0:test_slice.start]
    second_cross_term = pae_array[test_slice, test_slice.stop:]
    third_cross_term = pae_array[0:test_slice.start, test_slice]
    fourth_cross_term = pae_array[test_slice.stop:, test_slice]


    avg_intra_split_pae = (
                                  first_cross_term.sum()
                                  + second_cross_term.sum()
                                  + third_cross_term.sum()
                                  + fourth_cross_term.sum()
                          ) / (
                                  first_cross_term.numel()
                                  + second_cross_term.numel()
                                  + third_cross_term.numel()
                                  + fourth_cross_term.numel()
                          )

    pae_region_scale_factor = avg_intra_split_pae / (
            avg_inter_split_pae + 1e-9
    )

    return pae_region_scale_factor

# splits_rf2: list[int] = split_by_pae(pae_array_rf2, min_split_length=100)
# splits_af2: list[int] = split_by_pae(pae_array_af2, min_split_length=1)
pae_enrichment = compute_pae_enrichment(pae_array_af2)

# print(f"{splits_af2=}")
# print(f"{splits_rf2=}")

plt.contour(torch.nan_to_num(pae_enrichment))
plt.savefig(fname="pae_enrichment_af2.png")

# print(f"{len(pae_array_rf2)}=")
print(f"{len(pae_array_af2)}=")
# print(f"{inter_vs_intra_pae_score(pae_cumsum_rf2, slice(0, 400))}")
print(f"{inter_vs_intra_pae_score_naive(pae_cumsum_af2, slice(312, 450))}")
print(f"{pae_enrichment[312,450]}")

for j in range(len(pae_enrichment)):
    for i in range(j):
        test_slice = slice(i, j)
        assert torch.isclose(pae_enrichment[i, j], torch.scalar_tensor(inter_vs_intra_pae_score_naive(pae_array_af2, test_slice))), f"Failed for {i} {j}"

# print(pae_array_rf2.dtype)
