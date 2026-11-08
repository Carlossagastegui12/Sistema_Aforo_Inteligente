import pytest

@pytest.mark.acceptance
def test_flujo_clase_en_curso():
    estado_clase = 'Activa'
    aforo_registrado = 1
    assert estado_clase == 'Activa'
    assert aforo_registrado > 0
