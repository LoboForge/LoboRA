from lobora.cache import cache_key


def test_cache_key_includes_model_rev():
    a = cache_key(
        kind="latents",
        path="/data/a.mp4",
        caption="hello",
        height=832,
        width=480,
        frames_or_latent_t=7,
        model_rev="main",
    )
    b = cache_key(
        kind="latents",
        path="/data/a.mp4",
        caption="hello",
        height=832,
        width=480,
        frames_or_latent_t=7,
        model_rev="v2",
    )
    assert a != b
    assert len(a) == 64


def test_cache_key_changes_with_caption_and_size():
    base = dict(
        kind="text",
        path="/data/a.mp4",
        caption="hello",
        height=832,
        width=480,
        frames_or_latent_t=7,
        model_rev="main",
    )
    assert cache_key(**base) != cache_key(**{**base, "caption": "other"})
    assert cache_key(**base) != cache_key(**{**base, "height": 768})
