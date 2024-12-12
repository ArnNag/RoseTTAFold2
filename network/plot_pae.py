import numpy as np
import torch
import json
from icecream import ic
from matplotlib import pyplot as plt

from density import split_by_pae, compute_pae_enrichment

af2_pae_path = "/home/nagleam/Downloads/AF-T0D3N5-F1-predicted_aligned_error_v4.json"
pae_array_af2 = torch.tensor(json.load(open(af2_pae_path))[0]["predicted_aligned_error"], dtype=torch.float)

def two_d_cumsum(pae_array: torch.Tensor) -> torch.Tensor:
    pae_cumsum_rows = torch.cumsum(pae_array, dim=0)
    pae_cumsum = torch.cumsum(pae_cumsum_rows, dim=1)
    from torch.nn import functional

    pae_cumsum = functional.pad(input=pae_cumsum, pad=(1, 0, 1, 0), mode='constant', value=0.)
    return pae_cumsum

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
            avg_inter_split_pae
    )

    return pae_region_scale_factor

pae_enrichment = compute_pae_enrichment(pae_array_af2)

plt.contour(torch.nan_to_num(pae_enrichment))
plt.savefig(fname="pae_enrichment_af2.png")

print(f"{len(pae_array_af2)=}")
print(f"{inter_vs_intra_pae_score_naive(pae_array_af2, slice(911, 912))=}")
print(f"{0.5 * pae_enrichment[911, 912]=}")
print(f"{inter_vs_intra_pae_score_naive(pae_array_af2, slice(910, 913))=}")
print(f"{0.5 * pae_enrichment[910, 913]=}")

for j in range(len(pae_enrichment)):
    for i in range(j):
        test_slice = slice(i, j)
        fast_pae_enrichment = 0.5 * pae_enrichment[i, j]
        slow_pae_enrichment = torch.scalar_tensor(inter_vs_intra_pae_score_naive(pae_array_af2, test_slice))
        assert torch.isclose(fast_pae_enrichment, slow_pae_enrichment, atol=1e-1) or (fast_pae_enrichment > 1e7 and slow_pae_enrichment > 1e7) or ((j - i) == 1), f"Failed for {i} {j}. Fast: {fast_pae_enrichment}, slow: {slow_pae_enrichment}"
