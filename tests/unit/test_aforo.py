import pytest

def calcular_aforo_actual(personas_dentro, limite):
    return personas_dentro <= limite

@pytest.mark.unit
def test_verificar_limite_aforo():
    assert calcular_aforo_actual(15, 20) == True
    assert calcular_aforo_actual(25, 20) == False
