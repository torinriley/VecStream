import numpy as np
from hypothesis import given, settings, strategies as st

from vecstream.hnsw_index import HNSWIndex


@given(
    dimension=st.integers(min_value=1, max_value=12),
    count=st.integers(min_value=1, max_value=50),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=30, deadline=None)
def test_random_insert_delete_sequences_preserve_invariants(dimension, count, seed):
    rng = np.random.default_rng(seed)
    index = HNSWIndex(dimension, M=4, ef_construction=16, seed=seed)
    for i in range(count):
        index.add_item(str(i), rng.normal(size=dimension).astype(np.float32))
        index.validate()
    for i in rng.permutation(count)[: count // 2]:
        index.remove_item(str(i))
        index.validate()
