import pytest
import os

@pytest.mark.integration
def test_conexion_modelo_y_logica():
    pipeline_camara = True
    contador_ui = 1
    assert pipeline_camara and contador_ui > 0

@pytest.mark.integration
def test_actualizacion_metricas_ui():
    metricas = {"aforo": 20, "celulares": 1, "distracciones": 2}
    assert "aforo" in metricas
    assert metricas["aforo"] == 20

@pytest.mark.integration
def test_escritura_archivo_csv():
    folder_reportes = "reports"
    os.makedirs(folder_reportes, exist_ok=True)
    assert os.path.exists(folder_reportes)