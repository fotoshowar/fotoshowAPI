# FotoShow API — InsightFace CPU

Contenedor exclusivo buffalo_l, sin SFace ni OCR. HTTP /process con X-API-Key y TCP compatible con FotoShow.

Instalar Docker Engine y Compose. Clonar en /opt/fotoshow-api, copiar .env.example a .env, elegir clave y restaurar modelos buffalo_l del respaldo privado en /root/.insightface. Ejecutar ./instalar.sh. Puertos solo loopback: publicar HTTP mediante proxy HTTPS; TCP solo con red privada y firewall.

Desplegador: instalar github.com/fotoshowar/desplegador; agregar a /etc/desplegador/repos.conf la línea:

fotoshow-api | /opt/fotoshow-api | compose.yml | build | main

Ejecutar instalar.sh del desplegador para registrar el montaje. Cada merge a main reconstruye, verifica salud y revierte si falla.

FotoShow: AI_SERVER_URL=https://dominio-api y AI_SERVER_KEY. AI_WORKER_HOST=127.0.0.1 conserva local primero y API como respaldo. Para preferir el VPS dedicado usar su IP privada en AI_WORKER_HOST, puerto 54400, restringido por firewall.

Modelos y claves fuera de Git. Ver recognition/insightface/README.md.
