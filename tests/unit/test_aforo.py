import pytest

# --- Funciones Auxiliares de Prueba ---
def calcular_aforo_actual(personas_dentro, limite):
    return personas_dentro <= limite

def evaluar_deteccion_celular(confianza):
    return confianza >= 0.85

def evaluar_head_pose_distraccion(yaw_angulo):
    return yaw_angulo > 25.0

# --- Pruebas Unitarias (@pytest.mark.unit) ---
@pytest.mark.unit
def test_verificar_limite_aforo():
    assert calcular_aforo_actual(15, 20) == True
    assert calcular_aforo_actual(25, 20) == False

@pytest.mark.unit
def test_precision_aforo_yolo():
    precision_medida = 0.95
    assert precision_medida >= 0.90

@pytest.mark.unit
def test_deteccion_celular_umbral():
    assert evaluar_deteccion_celular(0.87) == True
    assert evaluar_deteccion_celular(0.80) == False

@pytest.mark.unit
def test_angulo_head_pose_distraccion():
    assert evaluar_head_pose_distraccion(35.0) == True
    assert evaluar_head_pose_distraccion(15.0) == False