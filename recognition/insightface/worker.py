"""
AI Worker — TCP server que corre InsightFace (reconocimiento facial) exclusivamente, con buffalo_l en CPU.

Protocolo: enviar JSON con {"action": "...", ...} terminado en newline (\n).
Respuesta: JSON + newline.

Acciones disponibles:
  {"action": "health"}
    → {"status": "ok", "models": ["insightface"]}

  {"action": "process_image", "image_b64": "<base64 JPEG/PNG>"}
    → {
        "faces": [
          {"embedding": [...512 floats], "bbox": {"x":int,"y":int,"w":int,"h":int}, "det_score": float}
        ],
        "bib_numbers": ["123", "456"],
        "faces_count": int
      }

  {"action": "compare_embedding", "embedding_a": [...], "embedding_b": [...]}
    → {"similarity": float, "match": bool}

Levantar: python ai_worker.py
"""
import asyncio
import base64
import concurrent.futures
import json
import hmac
import logging
import os
import signal
import sys
import threading
from io import BytesIO

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [AI Worker] %(message)s")
log = logging.getLogger(__name__)

HOST = os.getenv("AI_WORKER_BIND_HOST", "127.0.0.1")
PORT = int(os.getenv("AI_WORKER_PORT", "54400"))  # 54321 excluido por Hyper-V/Docker en Windows
HTTP_HOST = os.getenv("AI_WORKER_HTTP_BIND_HOST", "127.0.0.1")
HTTP_PORT = int(os.getenv("AI_WORKER_HTTP_PORT", "54401"))  # HTTP API para el navegador del fotógrafo

# InsightFace NO es thread-safe: un solo hilo a la vez para inferencia
_ai_lock     = threading.Lock()
_ai_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
API_KEY = os.getenv("INSIGHTFACE_API_KEY", "")
HTTP_MAX_BODY = 50 * 1024 * 1024  # 50 MB límite para imágenes en base64
FACE_SIMILARITY_THRESHOLD = 0.75  # cosine distance threshold for "same person"

# Lazy-loaded models
_face_app = None


def _load_insightface():
    global _face_app
    if _face_app is None:
        log.info("Loading InsightFace model (buffalo_l)…")
        import onnxruntime as ort
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = int(os.getenv("AI_THREADS", "1"))
        session_options.inter_op_num_threads = 1
        from insightface.app import FaceAnalysis
        # allowed_modules=['detection','recognition'] excluye 1k3d68/2d106det/genderage
        # que causan AttributeError en estimate_affine_matrix_3d23d cuando se compila.
        # Solo necesitamos bbox+det_score (detection) y embedding (recognition).
        _face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            sess_options=session_options,
            allowed_modules=["detection", "recognition"],
        )
        _face_app.prepare(ctx_id=-1, det_size=(640, 640))
        log.info("InsightFace ready ✓")
    return _face_app



def _bytes_to_cv2(image_bytes: bytes):
    import cv2
    from PIL import ImageOps
    # Aplicar rotación EXIF: cv2.imdecode ignora los metadatos EXIF,
    # PIL los lee correctamente y los aplica antes de convertir.
    pil_img = ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")
    arr = np.array(pil_img, dtype=np.uint8)
    # PIL devuelve RGB, cv2 espera BGR
    img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return img


def _detect_faces(image_bytes: bytes) -> list:
    """Return list of face dicts with embedding + bbox."""
    with _ai_lock:   # InsightFace no es thread-safe
        app = _load_insightface()
        img = _bytes_to_cv2(image_bytes)
        faces = app.get(img)
    results = []
    for face in faces:
        emb = face.embedding.tolist()  # 512-dim
        x1, y1, x2, y2 = face.bbox.astype(int)
        results.append({
            "embedding": emb,
            "bbox": {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)},
            "det_score": float(face.det_score),
        })
    return results



def _cosine_similarity(a: list, b: list) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def handle_request(payload: dict) -> dict:
    action = payload.get("action", "")

    if action == "health":
        return {"status": "ok", "models": ["insightface"]}

    elif action == "process_image":
        image_b64 = payload.get("image_b64", "")
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            return {"error": "invalid base64 image"}

        faces = _detect_faces(image_bytes)

        return {
            "faces": faces,
            "faces_count": len(faces),
        }

    elif action == "compare_embedding":
        a = payload.get("embedding_a", [])
        b = payload.get("embedding_b", [])
        sim = _cosine_similarity(a, b)
        return {
            "similarity": sim,
            "match": sim >= (1 - FACE_SIMILARITY_THRESHOLD),
        }

    else:
        return {"error": f"unknown action: {action}"}


async def _http_handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """
    Servidor HTTP mínimo (asyncio puro) para que el navegador del fotógrafo
    pueda enviar imágenes y recibir embeddings sin depender del servidor cloud.

    Rutas:
      GET  /health  → {"status": "ok"}
      POST /process → {"image_b64": "<base64>"} → {"faces": [...], "faces_count": N, "bib_numbers": [...]}
    """
    # Headers CORS que se incluyen en todas las respuestas
    CORS = (
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        # Necesario para Chrome cuando la página pública accede a localhost
        "Access-Control-Allow-Private-Network: true\r\n"
    )

    def _respond(status: str, body: str = "", ct: str = "application/json"):
        b = body.encode("utf-8")
        head = (
            f"HTTP/1.1 {status}\r\n"
            f"Content-Type: {ct}; charset=utf-8\r\n"
            f"{CORS}"
            f"Content-Length: {len(b)}\r\n"
            f"\r\n"
        ).encode("utf-8")
        writer.write(head + b)

    try:
        # ── Leer línea de petición (30s para la primera línea) ─────────────
        req_line = await asyncio.wait_for(reader.readline(), timeout=30.0)
        if not req_line:
            return
        parts = req_line.decode("utf-8", errors="replace").strip().split()
        if len(parts) < 2:
            return
        method, path = parts[0].upper(), parts[1]

        # ── Leer cabeceras (30s por línea — suficiente para cualquier red local)
        content_length = 0
        request_key = ""
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=30.0)
            if not line or line in (b"\r\n", b"\n"):
                break
            decoded = line.decode("utf-8", errors="replace").strip()
            if decoded.lower().startswith("x-api-key:"):
                request_key = decoded.split(":", 1)[1].strip()
            if decoded.lower().startswith("content-length:"):
                try:
                    content_length = int(decoded.split(":", 1)[1].strip())
                except ValueError:
                    pass

        # ── OPTIONS preflight ─────────────────────────────────────────────
        if method == "OPTIONS":
            writer.write(f"HTTP/1.1 204 No Content\r\n{CORS}\r\n".encode("utf-8"))
            await writer.drain()
            return

        # ── GET /health ───────────────────────────────────────────────────
        if method == "GET" and path == "/health":
            _respond("200 OK", json.dumps({"status": "ok", "models": ["insightface"]}))
            await writer.drain()
            return

        if API_KEY and not hmac.compare_digest(request_key, API_KEY):
            _respond("401 Unauthorized", json.dumps({"error": "API key inválida"}))
            await writer.drain()
            return

        # ── POST /process ─────────────────────────────────────────────────
        if method == "POST" and path == "/process":
            if content_length <= 0 or content_length > HTTP_MAX_BODY:
                _respond("400 Bad Request", json.dumps({"error": "content-length inválido o demasiado grande"}))
                await writer.drain()
                return

            log.info(f"POST /process — leyendo {content_length/1024:.0f} KB…")
            # readexactly garantiza leer todos los bytes del body (read() puede devolver menos)
            body_bytes = await asyncio.wait_for(reader.readexactly(content_length), timeout=120.0)
            try:
                payload = json.loads(body_bytes)
            except json.JSONDecodeError:
                _respond("400 Bad Request", json.dumps({"error": "JSON inválido"}))
                await writer.drain()
                return

            # POST /process siempre es process_image (el browser no envía action)
            payload.setdefault("action", "process_image")
            loop = asyncio.get_running_loop()
            log.info("POST /process — procesando con InsightFace…")
            try:
                result = await loop.run_in_executor(
                    _ai_executor,
                    lambda: handle_request(payload),
                )
            except Exception as e:
                log.exception("Error en HTTP /process")
                _respond("500 Internal Server Error", json.dumps({"error": str(e)}))
                await writer.drain()
                return

            log.info(f"POST /process — OK: {result.get('faces_count', 0)} cara(s)")
            _respond("200 OK", json.dumps(result))
            await writer.drain()
            return

        # ── 404 ───────────────────────────────────────────────────────────
        _respond("404 Not Found", json.dumps({"error": "ruta no encontrada"}))
        await writer.drain()

    except asyncio.TimeoutError:
        log.warning("HTTP handler timeout (conexión lenta o cliente desconectado)")
    except asyncio.IncompleteReadError:
        log.warning("HTTP handler: cliente cerró la conexión antes de enviar el body completo")
    except Exception:
        log.exception("HTTP handler error")
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info("peername")
    log.info(f"Connection from {addr}")
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                payload = json.loads(line.decode("utf-8").strip())
                response = await asyncio.get_running_loop().run_in_executor(
                    _ai_executor, handle_request, payload)
            except json.JSONDecodeError:
                response = {"error": "invalid JSON"}
            except Exception as e:
                log.exception("Error processing request")
                response = {"error": str(e)}

            writer.write((json.dumps(response) + "\n").encode("utf-8"))
            await writer.drain()
    except asyncio.IncompleteReadError:
        pass
    finally:
        writer.close()
        log.info(f"Connection closed: {addr}")


async def main():
    log.info("Pre-loading models…")
    try:
        _load_insightface()
    except Exception as e:
        log.exception("InsightFace failed to load")
        raise

    # limit=50MB para soportar imágenes grandes en base64
    tcp_server  = await asyncio.start_server(handle_client, HOST, PORT, limit=50 * 1024 * 1024)
    http_server = await asyncio.start_server(_http_handle, HTTP_HOST, HTTP_PORT, limit=50 * 1024 * 1024)

    log.info(f"TCP  AI Worker listening on {HOST}:{PORT}")
    log.info(f"HTTP AI Worker listening on {HTTP_HOST}:{HTTP_PORT}  (para el navegador)")

    loop = asyncio.get_running_loop()

    def shutdown():
        log.info("Shutting down…")
        tcp_server.close()
        http_server.close()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
            pass  # Windows

    async with tcp_server:
        async with http_server:
            await asyncio.gather(
                tcp_server.serve_forever(),
                http_server.serve_forever(),
            )


if __name__ == "__main__":
    asyncio.run(main())
