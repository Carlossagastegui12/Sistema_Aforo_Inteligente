import sys
import cv2
import csv
import os
from datetime import datetime
import warnings

# Ocultar las advertencias de PyTorch (FutureWarning) para mantener la terminal limpia
warnings.filterwarnings("ignore", category=FutureWarning)

# IMPORTANTE: ultralytics debe importarse ANTES de PyQt5 en Windows
from ultralytics import YOLO
modelo_objetos = YOLO('yolov8m.pt')

# Librerías de Reconocimiento Facial
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel,
    QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QFrame, QGridLayout
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer



# Librerías para gráficos
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# =========================================================
# HOJA DE ESTILOS GLOBAL (MODO OSCURO - DARK MODE)
# =========================================================
DARK_STYLESHEET = """
    QMainWindow { background-color: #0d1117; }
    QWidget { background-color: #0d1117; color: #e6edf3; font-family: 'Segoe UI', Arial, sans-serif; }
    QLabel { background-color: transparent; color: #e6edf3; }
    QPushButton { background-color: #1f6feb; color: #ffffff; font-size: 14px; font-weight: bold; padding: 12px 18px; border-radius: 8px; border: 1px solid #388bfd; }
    QPushButton:hover { background-color: #388bfd; border: 1px solid #58a6ff; }
    QPushButton:pressed { background-color: #1158c7; }
    QFrame { background-color: #161b22; border-radius: 12px; border: 1px solid #30363d; }
"""


# =========================================================
# 1. HILO DE VIDEO (Con Fusión de Reconocimiento Facial)
# =========================================================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    metricas_signal = pyqtSignal(int, int, int, int)

    def __init__(self):
        super().__init__()
        self.registro_acumulado = {}
        self.nombres_mapeados = {} # Diccionario para ligar ID_YOLO -> Nombre Real
        self._is_running = False
        self._is_paused = False

    def run(self):
        self._is_running = True
        
        # --- CARGA DE MODELOS YOLO ---
        modelo_pose = YOLO('yolov8n-pose.pt')
        modelo_objetos = YOLO('yolov8m.pt')
        
        # --- CARGA DE MODELOS FACIALES ---
        device = torch.device('cpu')
        # keep_all=False porque le pasaremos a MTCNN solo el recorte de 1 persona
        mtcnn = MTCNN(keep_all=False, device=device) 
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        # Cargar Base de Datos Visual
        carpeta_db = "alumnos_db"
        base_datos_rostros = {}
        if os.path.exists(carpeta_db):
            for archivo in os.listdir(carpeta_db):
                if archivo.endswith((".jpg", ".png", ".jpeg")):
                    ruta = os.path.join(carpeta_db, archivo)
                    nombre = os.path.splitext(archivo)[0]
                    img = Image.open(ruta).convert('RGB')
                    cara = mtcnn(img)
                    if cara is not None:
                        base_datos_rostros[nombre] = resnet(cara.unsqueeze(0).to(device)).detach()

        UMBRAL_DISTANCIA = 0.9
        cap = cv2.VideoCapture(0)
        UMBRAL_FRAMES = 25

        while self._is_running:
            if self._is_paused:
                self.msleep(100)
                continue

            exito, frame = cap.read()
            if not exito: continue
            
            # Formato RGB necesario tanto para PyQt5 como para FaceNet
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- DETECCIÓN DE CELULARES ---
            resultados_objetos = modelo_objetos(frame, conf=0.3, device='cpu', classes=[67], verbose=False)
            cajas_celulares = []
            for r in resultados_objetos:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cajas_celulares.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

            # --- TRACKING DE PERSONAS (ByteTrack) ---
            resultados = modelo_pose.track(frame, persist=True, conf=0.7, device='cpu', tracker="bytetrack.yaml")

            aforo_total = 0
            manos_levantadas_actual = 0
            alumnos_distraidos_actual = 0
            alumnos_celular_actual = 0
            
            # SOLUCIÓN DEL ERROR: Inicializamos frame_anotado antes del bucle
            frame_anotado = frame.copy()

            for resultado in resultados:
                # Ahora sí, dibujamos de forma segura
                frame_anotado = resultado.plot()
                aforo_total = len(resultado.boxes)

                if (resultado.keypoints is not None and len(resultado.keypoints.xy) > 0 and resultado.boxes.id is not None):
                    keypoints_personas = resultado.keypoints.xy
                    ids = resultado.boxes.id.int().cpu().tolist()

                    for i, kp in enumerate(keypoints_personas):
                        if i < len(ids):
                            id_persona = ids[i]

                            # 1. INICIALIZACIÓN DE REGISTRO
                            if id_persona not in self.registro_acumulado:
                                hora_llegada = datetime.now().strftime("%H:%M:%S")
                                self.registro_acumulado[id_persona] = {
                                    "hora_llegada": hora_llegada,
                                    "participaciones": 0, "distracciones": 0, "uso_celular": 0,
                                    "frames_consecutivos_mano": 0, "frames_consecutivos_distraido": 0, "frames_consecutivos_celular": 0,
                                    "mano_contada": False, "distraccion_contada": False, "celular_contado": False
                                }
                                self.nombres_mapeados[id_persona] = f"Estudiante {id_persona}"

                            # 2. INTENTO DE RECONOCIMIENTO FACIAL (Lógica Lazy)
                            if self.nombres_mapeados[id_persona].startswith("Estudiante"):
                                px1, py1, px2, py2 = [int(v) for v in resultado.boxes.xyxy[i].tolist()]
                                h_f, w_f = frame_rgb.shape[:2]
                                px1, py1 = max(0, px1), max(0, py1)
                                px2, py2 = min(w_f, px2), min(h_f, py2)
                                
                                recorte = frame_rgb[py1:py2, px1:px2]
                                if recorte.size > 0:
                                    img_pil = Image.fromarray(recorte)
                                    cara_recortada = mtcnn(img_pil)
                                    
                                    if cara_recortada is not None:
                                        huella_vivo = resnet(cara_recortada.unsqueeze(0).to(device)).detach()
                                        distancia_min = float('inf')
                                        nombre_detectado = None
                                        
                                        for nombre_db, huella_db in base_datos_rostros.items():
                                            distancia = torch.dist(huella_vivo, huella_db).item()
                                            if distancia < distancia_min and distancia < UMBRAL_DISTANCIA:
                                                distancia_min = distancia
                                                nombre_detectado = nombre_db
                                                
                                        if nombre_detectado:
                                            self.nombres_mapeados[id_persona] = nombre_detectado

                            # 3. LÓGICA DE POSTURAS Y CELULAR
                            if len(kp) >= 11:
                                nariz_y = float(kp[0][1])
                                h_promedio_y = (float(kp[5][1]) + float(kp[6][1])) / 2
                                mano_arriba = (float(kp[9][1]) < float(kp[5][1]) or float(kp[10][1]) < float(kp[6][1]))
                                distraido = ((h_promedio_y - nariz_y) < 30)

                                usando_celular = False
                                if len(cajas_celulares) > 0:
                                    px1, py1, px2, py2 = resultado.boxes.xyxy[i].tolist()
                                    for (cx1, cy1, cx2, cy2) in cajas_celulares:
                                        if not (cx2 < px1 or cx1 > px2 or cy2 < py1 or cy1 > py2):
                                            usando_celular = True
                                            break

                                if mano_arriba: manos_levantadas_actual += 1
                                if distraido: alumnos_distraidos_actual += 1
                                if usando_celular: alumnos_celular_actual += 1

                                # Antirrebote Participación
                                if mano_arriba:
                                    self.registro_acumulado[id_persona]["frames_consecutivos_mano"] += 1
                                    if (self.registro_acumulado[id_persona]["frames_consecutivos_mano"] >= UMBRAL_FRAMES 
                                        and not self.registro_acumulado[id_persona]["mano_contada"]):
                                        self.registro_acumulado[id_persona]["participaciones"] += 1
                                        self.registro_acumulado[id_persona]["mano_contada"] = True
                                else:
                                    self.registro_acumulado[id_persona]["frames_consecutivos_mano"] = 0
                                    self.registro_acumulado[id_persona]["mano_contada"] = False

                                # Antirrebote Distracción
                                if distraido:
                                    self.registro_acumulado[id_persona]["frames_consecutivos_distraido"] += 1
                                    if (self.registro_acumulado[id_persona]["frames_consecutivos_distraido"] >= UMBRAL_FRAMES 
                                        and not self.registro_acumulado[id_persona]["distraccion_contada"]):
                                        self.registro_acumulado[id_persona]["distracciones"] += 1
                                        self.registro_acumulado[id_persona]["distraccion_contada"] = True
                                else:
                                    self.registro_acumulado[id_persona]["frames_consecutivos_distraido"] = 0
                                    self.registro_acumulado[id_persona]["distraccion_contada"] = False

                                # Antirrebote Celular
                                if usando_celular:
                                    self.registro_acumulado[id_persona]["frames_consecutivos_celular"] += 1
                                    if (self.registro_acumulado[id_persona]["frames_consecutivos_celular"] >= UMBRAL_FRAMES 
                                        and not self.registro_acumulado[id_persona]["celular_contado"]):
                                        self.registro_acumulado[id_persona]["uso_celular"] += 1
                                        self.registro_acumulado[id_persona]["celular_contado"] = True
                                else:
                                    self.registro_acumulado[id_persona]["frames_consecutivos_celular"] = 0
                                    self.registro_acumulado[id_persona]["celular_contado"] = False

                                # DIBUJAR NOMBRE REAL EN EL FRAME
                                nombre_mostrar = self.nombres_mapeados[id_persona]
                                bx1, by1, bx2, by2 = [int(v) for v in resultado.boxes.xyxy[i].tolist()]
                                # Etiqueta con el nombre real
                                cv2.putText(frame_anotado, nombre_mostrar, (bx1, by1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Enviar señal a Interfaz
            rgb_final = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_final.shape
            convert_to_Qt_format = QImage(rgb_final.data, w, h, ch * w, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)

            self.change_pixmap_signal.emit(p)
            self.metricas_signal.emit(aforo_total, manos_levantadas_actual, alumnos_distraidos_actual, alumnos_celular_actual)

        cap.release()

    def stop(self):
        self._is_running = False
        self.wait()

    def pause(self):
        self._is_paused = not self._is_paused


# =========================================================
# 2. CLASE DEL GRÁFICO (MODO OSCURO)
# =========================================================
class CanvasGrafico(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#161b22')
        self.axes = fig.add_subplot(111)
        self.axes.set_facecolor('#0d1117')
        super().__init__(fig)
        self.setStyleSheet("background-color: #161b22; border-radius: 8px;")
        self._dibujar(0, 0, 0)

    def _dibujar(self, total_p, total_d, total_c):
        self.axes.clear()
        self.axes.set_facecolor('#0d1117')

        categorias = ['Participaciones', 'Distracciones', 'Celular']
        valores    = [total_p, total_d, total_c]
        colores    = ['#3fb950', '#f85149', '#e3b341']

        bars = self.axes.bar(categorias, valores, color=colores, width=0.5, zorder=3)

        for bar, val in zip(bars, valores):
            self.axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(val), ha='center', va='bottom', color='#e6edf3', fontsize=11, fontweight='bold')

        self.axes.set_title('Historial Acumulado de la Clase', color='#e6edf3', fontsize=12, fontweight='bold', pad=10)
        self.axes.tick_params(colors='#8b949e')
        for spine in self.axes.spines.values(): spine.set_edgecolor('#30363d')

        max_val = max(valores) if max(valores) > 0 else 5
        self.axes.set_ylim(0, max_val + max(2, int(max_val * 0.2)))
        self.axes.yaxis.grid(True, color='#21262d', linewidth=0.8, zorder=0)
        self.axes.set_axisbelow(True)

        self.figure.tight_layout()
        self.draw()

    def actualizar_grafico(self, total_participaciones, total_distracciones, total_celulares):
        self._dibujar(total_participaciones, total_distracciones, total_celulares)


# =========================================================
# 3. TARJETA DE MÉTRICA (Diseño Premium)
# =========================================================
def crear_tarjeta_metrica(parent, titulo, valor_inicial, color_valor):
    frame = QFrame(parent)
    frame.setFixedHeight(90) # Un poco más altas para que respiren
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(5)

    lbl_titulo = QLabel(titulo, frame)
    lbl_titulo.setAlignment(Qt.AlignCenter) # Texto centrado
    lbl_titulo.setStyleSheet("font-size: 12px; color: #8b949e; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; background: transparent; border: none;")

    lbl_valor = QLabel(valor_inicial, frame)
    lbl_valor.setAlignment(Qt.AlignCenter) # Texto centrado
    lbl_valor.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {color_valor}; background: transparent; border: none;")

    layout.addWidget(lbl_titulo)
    layout.addWidget(lbl_valor)
    return frame, lbl_valor


# =========================================================
# 4. INTERFAZ PRINCIPAL
# =========================================================
class DashboardAnalitico(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema Inteligente UPN - Dashboard Analítico")
        # 1. Agrandamos la ventana principal (de 1180x680 a 1360x768)
        self.setGeometry(50, 50, 1360, 768) 
        self.setStyleSheet(DARK_STYLESHEET)

        widget_central = QWidget()
        layout_principal = QHBoxLayout()
        layout_principal.setContentsMargins(20, 20, 20, 20)
        layout_principal.setSpacing(20)

        # PANEL IZQUIERDO -> VIDEO (Se mantiene igual)
        frame_video = QFrame()
        frame_video.setFixedSize(648, 488)
        layout_video = QVBoxLayout(frame_video)
        layout_video.setContentsMargins(4, 4, 4, 4)

        self.label_video = QLabel()
        self.label_video.setFixedSize(640, 480)
        self.label_video.setAlignment(Qt.AlignCenter)
        self.label_video.setStyleSheet("background-color: #010409; border-radius: 8px;")
        layout_video.addWidget(self.label_video)
        layout_principal.addWidget(frame_video)

        # PANEL DERECHO -> MÉTRICAS Y GRÁFICO
        layout_derecho = QVBoxLayout()
        layout_derecho.setSpacing(15)

        label_titulo = QLabel("⬤  Análisis Pedagógico en Tiempo Real")
        label_titulo.setStyleSheet("font-size: 18px; font-weight: bold; color: #58a6ff; padding: 5px 0px; border: none; background: transparent;")
        layout_derecho.addWidget(label_titulo)

        # CREACIÓN DE TARJETAS
        tarjeta_aforo, self.label_aforo = crear_tarjeta_metrica(widget_central, "Aforo Actual", "0 personas", "#e6edf3")
        tarjeta_manos, self.label_manos = crear_tarjeta_metrica(widget_central, "Manos Levantadas", "0", "#3fb950")
        tarjeta_distraidos, self.label_distraidos = crear_tarjeta_metrica(widget_central, "Posibles Distraídos", "0", "#f85149")
        tarjeta_celular, self.label_celular = crear_tarjeta_metrica(widget_central, "Usando Celular 📱", "0", "#e3b341")

        # 2. NUEVO DISEÑO EN CUADRÍCULA (Grid 2x2)
        layout_tarjetas = QGridLayout()
        layout_tarjetas.setSpacing(12)
        layout_tarjetas.addWidget(tarjeta_aforo, 0, 0)       # Fila 0, Columna 0
        layout_tarjetas.addWidget(tarjeta_manos, 0, 1)       # Fila 0, Columna 1
        layout_tarjetas.addWidget(tarjeta_distraidos, 1, 0)  # Fila 1, Columna 0
        layout_tarjetas.addWidget(tarjeta_celular, 1, 1)     # Fila 1, Columna 1
        
        layout_derecho.addLayout(layout_tarjetas)

        # 3. GRÁFICO (Ahora tiene proporciones mucho más amplias)
        self.canvas = CanvasGrafico(widget_central, width=6, height=4.5)
        layout_derecho.addWidget(self.canvas)

        # CONTROLES
        layout_controles = QHBoxLayout()
        self.btn_iniciar = QPushButton("▶ Iniciar Clase")
        self.btn_iniciar.setStyleSheet("background-color: #238636; color: white; font-weight: bold;")
        self.btn_iniciar.clicked.connect(self.iniciar_camara)
        
        self.btn_pausa = QPushButton("⏸ Pausar")
        self.btn_pausa.setStyleSheet("background-color: #e3b341; color: black; font-weight: bold;")
        self.btn_pausa.setEnabled(False)
        self.btn_pausa.clicked.connect(self.pausar_camara)

        self.btn_cerrar = QPushButton("⏹ Cerrar")
        self.btn_cerrar.setStyleSheet("background-color: #da3633; color: white; font-weight: bold;")
        self.btn_cerrar.clicked.connect(self.cerrar_programa)

        layout_controles.addWidget(self.btn_iniciar)
        layout_controles.addWidget(self.btn_pausa)
        layout_controles.addWidget(self.btn_cerrar)
        layout_derecho.addLayout(layout_controles)

        # BOTÓN EXPORTAR
        self.btn_exportar = QPushButton("⬇  Exportar Reporte Final (CSV)")
        self.btn_exportar.setFixedHeight(44)
        self.btn_exportar.clicked.connect(self.guardar_datos_csv)
        layout_derecho.addWidget(self.btn_exportar)

        layout_principal.addLayout(layout_derecho)
        widget_central.setLayout(layout_principal)
        self.setCentralWidget(widget_central)

        # HILO DE VIDEO
        self.hilo_video = VideoThread()
        self.hilo_video.change_pixmap_signal.connect(self.actualizar_imagen)
        self.hilo_video.metricas_signal.connect(self.actualizar_metricas)

        # TIMER DEL GRÁFICO
        self.timer_grafico = QTimer()
        self.timer_grafico.timeout.connect(self.refrescar_estadisticas)
        self.timer_grafico.start(2000)

    def iniciar_camara(self):
        if not self.hilo_video.isRunning():
            self.hilo_video.start()
            self.btn_iniciar.setEnabled(False)
            self.btn_pausa.setEnabled(True)
            self.btn_iniciar.setText("Clase en Curso...")
            self.btn_iniciar.setStyleSheet("background-color: #2ea043; color: white; font-weight: bold;")

    def pausar_camara(self):
        self.hilo_video.pause()
        if self.hilo_video._is_paused:
            self.btn_pausa.setText("▶ Reanudar")
            self.btn_pausa.setStyleSheet("background-color: #2ea043; color: white; font-weight: bold;")
        else:
            self.btn_pausa.setText("⏸ Pausar")
            self.btn_pausa.setStyleSheet("background-color: #e3b341; color: black; font-weight: bold;")

    def cerrar_programa(self):
        if self.hilo_video.isRunning():
            self.hilo_video.stop()
        self.close()

    def actualizar_imagen(self, cv_img):
        self.label_video.setPixmap(QPixmap.fromImage(cv_img))

    def actualizar_metricas(self, aforo, manos, distraidos, celulares):
        self.label_aforo.setText(f"{aforo} personas")
        self.label_manos.setText(str(manos))
        self.label_distraidos.setText(str(distraidos))
        self.label_celular.setText(str(celulares))

    def refrescar_estadisticas(self):
        registros = self.hilo_video.registro_acumulado.values()
        total_p = sum(d["participaciones"] for d in registros)
        total_d = sum(d["distracciones"]   for d in registros)
        total_c = sum(d["uso_celular"]     for d in registros)
        self.canvas.actualizar_grafico(total_p, total_d, total_c)

    def guardar_datos_csv(self):
        nombre_archivo = "reporte_analitico.csv"
        archivo_existe = os.path.isfile(nombre_archivo) and os.stat(nombre_archivo).st_size > 0

        # =========================================================
        # CONSOLIDACIÓN DE DATOS (Agrupación por Nombre)
        # =========================================================
        datos_consolidados = {}

        for id_est, datos in self.hilo_video.registro_acumulado.items():
            # Obtenemos el nombre (ej: "fabrizio" o "Estudiante 4")
            nombre_alumno = self.hilo_video.nombres_mapeados.get(id_est, f"Estudiante {id_est}")

            if nombre_alumno not in datos_consolidados:
                # Si es la primera vez que vemos este nombre, copiamos sus datos
                datos_consolidados[nombre_alumno] = {
                    "hora_llegada": datos["hora_llegada"],
                    "participaciones": datos["participaciones"],
                    "distracciones": datos["distracciones"],
                    "uso_celular": datos["uso_celular"]
                }
            else:
                # Si el alumno ya estaba registrado (salió y volvió a entrar)
                # Sumamos sus métricas acumuladas
                datos_consolidados[nombre_alumno]["participaciones"] += datos["participaciones"]
                datos_consolidados[nombre_alumno]["distracciones"] += datos["distracciones"]
                datos_consolidados[nombre_alumno]["uso_celular"] += datos["uso_celular"]

                # Comparamos las horas para quedarnos siempre con la primera (la más antigua)
                hora_actual_guardada = datetime.strptime(datos_consolidados[nombre_alumno]["hora_llegada"], "%H:%M:%S")
                hora_nueva = datetime.strptime(datos["hora_llegada"], "%H:%M:%S")

                if hora_nueva < hora_actual_guardada:
                    datos_consolidados[nombre_alumno]["hora_llegada"] = datos["hora_llegada"]

        # =========================================================
        # ESCRITURA EN EL ARCHIVO CSV
        # =========================================================
        with open(nombre_archivo, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if not archivo_existe:
                writer.writerow([
                    "Fecha Reporte", "Hora de Llegada", "Estudiante", 
                    "Participaciones", "Distracciones", "Uso de Celular"
                ])

            fecha_reporte = datetime.now().strftime("%Y-%m-%d")

            # Ahora iteramos sobre nuestro diccionario limpio y consolidado
            for nombre, metricas in datos_consolidados.items():
                writer.writerow([
                    fecha_reporte,
                    metricas["hora_llegada"],
                    nombre,
                    metricas["participaciones"],
                    metricas["distracciones"],
                    metricas["uso_celular"]
                ])

        # Retroalimentación visual del botón
        self.btn_exportar.setText("✔  ¡Reporte Guardado!")
        self.btn_exportar.setStyleSheet("background-color: #238636; color: white; font-weight: bold; padding: 12px; border-radius: 8px;")
        QTimer.singleShot(1500, lambda: (
            self.btn_exportar.setText("⬇  Exportar Reporte Final (CSV)"),
            self.btn_exportar.setStyleSheet("background-color: #1f6feb; color: white; font-weight: bold; padding: 12px; border-radius: 8px; border: 1px solid #388bfd;")
        ))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ventana = DashboardAnalitico()
    ventana.show()
    sys.exit(app.exec_())