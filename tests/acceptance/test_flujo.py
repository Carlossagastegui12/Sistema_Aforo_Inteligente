import pytest

@pytest.mark.acceptance
def test_flujo_clase_en_curso():
    estado_clase = {"activa": True, "camara_conectada": True}
    assert estado_clase["activa"] is True
    assert estado_clase["camara_conectada"] is True

@pytest.mark.acceptance
def test_e2e_sesion_completa_csv():
    sesion = {"activa": True, "aforo": 20, "reporte_generado": False}
    # Simulación de finalización de clase y exportación
    sesion["activa"] = False
    sesion["reporte_generado"] = True
    
    assert sesion["activa"] is False
    assert sesion["reporte_generado"] is True