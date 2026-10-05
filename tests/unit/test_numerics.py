import numpy as np
import pytest

from vecstream.errors import DimensionMismatchError, InvalidVectorError
from vecstream.vector_store import VectorStore


@pytest.mark.parametrize("bad", [[], [[1.0]], [1.0, np.nan], [1.0, np.inf], ["x"]])
def test_rejects_malformed_vectors(bad):
    with pytest.raises(InvalidVectorError):
        VectorStore().add_vector("bad", bad)


def test_dimension_mismatch_is_explicit():
    store = VectorStore()
    store.add_vector("a", [1, 2])
    with pytest.raises(DimensionMismatchError):
        store.search_similar([1, 2, 3])


def test_zero_vector_similarity_is_zero():
    store = VectorStore()
    store.add_vector("zero", [0, 0])
    store.add_vector("unit", [1, 0])
    assert dict(store.exact_search([0, 0], k=2, threshold=-1)) == {"zero": 0.0, "unit": 0.0}
