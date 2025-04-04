import pytest
import unipercept.utils.registry


@pytest.mark.parametrize("exist_ok", [True, False])
def test_anonymous_registry(exist_ok: bool):
    reg = unipercept.utils.registry.AnonymousRegistry[int](
        exist_ok=exist_ok, check=lambda x: x > 0
    )
    reg.add(1)
    assert reg.exist_ok == exist_ok
    assert 1 in reg
    assert len(reg) == 1

    # Test index exists
    if not exist_ok:
        with pytest.raises(unipercept.utils.registry.RegistryIndexExistsError):
            reg.add(1)
    else:
        reg.add(1)
        assert 1 in reg
        assert len(reg) == 1

    # Test registry value check
    with pytest.raises(unipercept.utils.registry.RegistryValueCheckError):
        reg.add(-1)
    assert len(reg) == 1
    reg.add(-2, skip_check=True)
    assert -2 in reg
    assert len(reg) == 2


@pytest.mark.parametrize("exist_ok", [True, False])
def test_indexed_registry(exist_ok: bool):
    reg = unipercept.utils.registry.IndexedRegistry[int](
        exist_ok=exist_ok, check=lambda x: x > 0
    )
    reg["foo"] = 42
    assert reg.exist_ok == exist_ok
    assert reg["foo"] == 42
    assert len(reg) == 1

    # Test index exists
    if not exist_ok:
        with pytest.raises(unipercept.utils.registry.RegistryIndexExistsError):
            reg["foo"] = 43
    else:
        reg["foo"] = 43
        assert reg["foo"] == 43
        assert len(reg) == 1

    # Test registry value check
    with pytest.raises(unipercept.utils.registry.RegistryValueCheckError):
        reg["bar"] = -1
    assert len(reg) == 1
    reg.register("bar", skip_check=True)(-2)
    assert reg["bar"] == -2
    assert len(reg) == 2
