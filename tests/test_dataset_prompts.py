from dataclasses import dataclass

from lobora.sampling import pick_prompts_from_dataset


@dataclass
class S:
    sample_id: str
    caption: str
    kind: str = "video"


def test_pick_from_dataset_is_seeded_and_sized():
    samples = [S(f"v{i}.mp4", f"caption number {i} with enough words", "video") for i in range(20)]
    a = pick_prompts_from_dataset(samples, n=4, seed=7)
    b = pick_prompts_from_dataset(samples, n=4, seed=7)
    c = pick_prompts_from_dataset(samples, n=4, seed=8)
    assert len(a) == 4
    assert [x["source_id"] for x in a] == [x["source_id"] for x in b]
    assert [x["source_id"] for x in a] != [x["source_id"] for x in c]
    # Do not assert caption text content — privacy.
    assert all("prompt" in x and "name" in x for x in a)
