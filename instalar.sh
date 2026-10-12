#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v docker >/dev/null || { echo "Instalá Docker Engine y Compose primero"; exit 1; }
[ -f .env ] || { cp .env.example .env; echo "Configurá .env y restaurá buffalo_l"; exit 1; }
./deploy/desplegar.sh
