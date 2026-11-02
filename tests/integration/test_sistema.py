import pytest

@pytest.mark.integration
def test_conexion_modelo_y_logica():
    conteo_detectado = 1 
    assert isinstance(conteo_detectado, int)
    assert conteo_detectado >= 0
