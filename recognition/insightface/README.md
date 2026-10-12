# InsightFace CPU reutilizable
Solo detección y reconocimiento buffalo_l; sin OCR ni SFace, sin claves ni base de datos.
Las versiones y el modelo son los mismos que en facerec: embeddings de 512 dimensiones compatibles con pgvector.
El modelo debe estar en /root/.insightface/models/buffalo_l (det_10g.onnx y w600k_r50.onnx).
Se monta solo lectura. Restaurar desde respaldo antes de iniciar.
Arranque independiente: docker compose -f recognition/insightface/compose.yml up -d --build
FotoShow: docker compose -f deploy/compose.yml up -d --build insightface
Configurar AI_WORKER_HOST=127.0.0.1 y AI_WORKER_PORT=54400 en .env.
Para volver a una API HTTP, configurar AI_SERVER_URL y AI_SERVER_KEY: local sigue siendo primera opción y API último recurso, según ai_service.py.
Puertos TCP 54400 y HTTP 54401 solo en loopback. No exponer públicamente.
En otro VPS usar INSIGHTFACE_BIND_IP con una IP privada y firewall restringido al cliente (TCP sin autenticación).
INSIGHTFACE_MODELS_DIR cambia la carpeta de modelos. CPU 0.75, memoria 850 MB, un hilo de inferencia.
Si el modelo no carga el proceso falla: health no oculta errores de arranque.
Los modelos y datos no se guardan en Git.
