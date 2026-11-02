import cv2
import torch
import os
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# ==========================================
# 1. INICIALIZAR LOS MODELOS DE RECONOCIMIENTO
# ==========================================
print("[INFO] Cargando modelos de IA Facial...")

# Dispositivo optimizado para CPU
device = torch.device('cpu')

# MTCNN: Detecta dónde está la cara y la recorta
mtcnn = MTCNN(keep_all=True, device=device)

# InceptionResnetV1: Extrae la "huella digital" del rostro (Embeddings)
# Usamos los pesos pre-entrenados de 'vggface2'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ==========================================
# 2. CARGAR LA BASE DE DATOS DE ALUMNOS
# ==========================================
carpeta_db = "alumnos_db"
base_datos_rostros = {}

print(f"[INFO] Leyendo fotos de la carpeta '{carpeta_db}'...")

if not os.path.exists(carpeta_db):
    os.makedirs(carpeta_db)
    print(f"[ADVERTENCIA] Se creó la carpeta '{carpeta_db}'. Por favor, añade fotos.")

# Recorremos todas las fotos en la carpeta
for archivo in os.listdir(carpeta_db):
    if archivo.endswith((".jpg", ".png", ".jpeg")):
        ruta_imagen = os.path.join(carpeta_db, archivo)
        nombre_alumno = os.path.splitext(archivo)[0] # Saca el nombre sin el .jpg (Ej: Fabrizio)
        
        # Abrimos la imagen y detectamos la cara
        img = Image.open(ruta_imagen).convert('RGB')
        cara_recortada = mtcnn(img)
        
        if cara_recortada is not None:
            # Calculamos la huella digital (embedding)
            # cara_recortada[0] porque asumimos que hay 1 sola cara en la foto de perfil
            huella = resnet(cara_recortada[0].unsqueeze(0).to(device))
            base_datos_rostros[nombre_alumno] = huella.detach()
            print(f"  -> Rostro de '{nombre_alumno}' registrado exitosamente.")
        else:
            print(f"  -> [ERROR] No se detectó ninguna cara en '{archivo}'.")

# ==========================================
# 3. PRUEBA EN TIEMPO REAL CON LA WEBCAM
# ==========================================
print("\n[INFO] Iniciando cámara... Presiona 'q' para salir.")
cap = cv2.VideoCapture(0)

# Umbral de tolerancia: Qué tan estricta es la comparación (0.8 es un buen estándar)
UMBRAL_DISTANCIA = 0.9

while True:
    exito, frame = cap.read()
    if not exito:
        break

    # Convertimos el frame de OpenCV (BGR) al formato PIL (RGB) que usa MTCNN
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)

    # Detectamos las caras en el frame en vivo
    cajas_caras, _ = mtcnn.detect(img_pil)
    caras_recortadas = mtcnn(img_pil)

    if caras_recortadas is not None and cajas_caras is not None:
        
        # Recorremos cada cara detectada en la cámara
        for i, cara in enumerate(caras_recortadas):
            # Extraemos la huella de la cara en vivo
            huella_vivo = resnet(cara.unsqueeze(0).to(device)).detach()
            
            nombre_detectado = "Desconocido"
            distancia_minima = float('inf')

            # Comparamos con todas las caras de nuestra base de datos
            for nombre_db, huella_db in base_datos_rostros.items():
                # Calculamos la distancia Euclidiana entre las huellas
                distancia = torch.dist(huella_vivo, huella_db).item()
                
                if distancia < distancia_minima:
                    distancia_minima = distancia
                    if distancia < UMBRAL_DISTANCIA:
                        nombre_detectado = nombre_db

            # Dibujamos el recuadro y el nombre
            box = cajas_caras[i]
            x1, y1, x2, y2 = [int(b) for b in box]
            
            # Cambia a verde si te reconoce, rojo si eres desconocido
            color = (0, 255, 0) if nombre_detectado != "Desconocido" else (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{nombre_detectado} ({distancia_minima:.2f})", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Prueba de Reconocimiento Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()