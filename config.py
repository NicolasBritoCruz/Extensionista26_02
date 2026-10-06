"""
Configuracao central do EPI Detect.

Todos os limites aqui foram calibrados em video real, nao escolhidos por
intuicao. Os comentarios registram a medicao que justifica cada valor,
para que ajustes futuros partam de dado e nao de tentativa.

Para adaptar a uma maquina ou camera especifica, nao edite este arquivo:
crie um settings_local.json na raiz do projeto com as chaves que quiser
sobrescrever. Ele e ignorado pelo git.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).parent

# ── fonte de video ────────────────────────────────────────────────────────────
# Aceita indice de webcam (0, 1, ...) ou URL RTSP de camera IP.
# Exemplo RTSP: "rtsp://usuario:senha@192.168.0.50:554/stream1"
SOURCE = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# ── modelo ────────────────────────────────────────────────────────────────────
# yolov8n-pose e o menor da familia. Baixa sozinho na primeira execucao.
MODEL_PATH = os.getenv("EPI_MODEL_PATH", "yolov8n-pose.pt")

# Resolucao de entrada do modelo. Nao existe valor bom para todo caso:
#   plano aberto, pessoas a ~150 px  -> 1280 (em 896 pessoas desaparecem)
#   plano fechado, pessoa a ~440 px  ->  768 (acerta mais e roda ~3x mais rapido)
# O YOLO responde por cerca de 99% do custo por frame, entao esse numero e
# o que mais mexe no desempenho. Ver detector.pick_input_size().
INPUT_SIZE = 1280
INPUT_SIZE_NEAR = 768

# Fracao da altura do quadro a partir da qual a cena e considerada plano
# fechado e passa a usar INPUT_SIZE_NEAR.
NEAR_SCENE_RATIO = 0.30

# ── limites de deteccao ───────────────────────────────────────────────────────
# 0.20 mantem pessoas ao fundo. Abaixo de 0.15 comecam a entrar caixas
# achatadas de reflexo no piso, que MAX_ASPECT descarta.
CONF_THRESH = 0.20

# Pessoa e sempre mais alta que larga, mesmo sentada. Reflexo no piso chega
# em proporcoes de 6:1, entao 2.6 separa os dois casos com folga.
MAX_ASPECT = 2.60

# Pose coerente tem 10 ou mais keypoints confiaveis; deteccao fantasma
# costuma trazer 1 ou 2.
MIN_KEYPOINTS = 4
KEYPOINT_MIN_CONF = 0.15

# Cobertura de cor de colete no tronco. Colete refletivo cobre bem mais
# que 9% da faixa; o valor baixo tolera oclusao parcial por carga ou volante.
VEST_THRESH = 0.09

# Cobertura de capacete na calota da cabeca. Medido em video real:
# com capacete a cobertura fica entre 0.29 e 0.82; cabeca descoberta da
# 0.00 depois de excluir pele e cabelo. O corte em 0.24 separa com folga.
HELMET_THRESH = 0.24

# ── modelo treinado de EPI ────────────────────────────────────────────────────
# Metodo de decisao do colete e do capacete:
#   "cor"    analise de cor e forma nos recortes da pose (metodo original)
#   "modelo" detector YOLO treinado para EPI (ver treino/treinar_epi.ipynb)
#   "clip"   CLIP pre-treinado classifica os recortes de cabeca e tronco
#            (capacete x bone x nada; colete x roupa), sem treino
#   "auto"   clip se o open_clip estiver instalado; senao o modelo, se o
#            arquivo de pesos existir; senao cor
# Em teste ao vivo o clip foi o unico que separou bone de capacete de forma
# consistente, inclusive trocando de ambiente, por isso e o preferido.
# A analise de cor confunde bone liso com capacete e camiseta laranja com
# colete; um modelo treinado com esses casos como negativos separa os dois.
EPI_METHOD = os.getenv("EPI_METHOD", "auto")
PPE_MODEL_PATH = os.getenv("EPI_PPE_MODEL", str(ROOT / "models" / "epi.pt"))
PPE_CONF = 0.35
PPE_IMGSZ = 640

# CLIP: ViT-B/32 com os pesos da OpenAI (~350 MB, baixados na primeira
# execucao). A variante quickgelu e a que corresponde a esses pesos.
CLIP_MODEL = "ViT-B-32-quickgelu"
CLIP_PRETRAINED = "openai"

# Abaixo destes tamanhos (altura do recorte, em pixels do quadro) ou desta
# margem entre a classe vencedora e a segunda, o EPI fica "nao avaliado"
# em vez de virar um chute. Medido com treino/avaliar.py em fotos de obra
# (EPI real) e em video proprio (bone, sem EPI):
#   margem 0.15     erros de bone vermelho como capacete: 7 -> 1 em 442
#   cabeca >= 50px  alarmes falsos de capacete em obra: 4 -> 2
#   tronco          o minimo de tamanho nao tirou nenhum erro, so gerava
#                   "nao avaliado" na webcam, onde o tronco aparece pouco
CLIP_MIN_HEAD_PX = 50
CLIP_MIN_TORSO_PX = 0
CLIP_MIN_MARGIN = 0.15

# Fracao da caixa do EPI que precisa cair dentro da regiao da pessoa
# (cabeca para capacete, tronco para colete) para ser atribuida a ela.
PPE_MIN_OVERLAP = 0.40

# ── estabilizacao temporal ────────────────────────────────────────────────────
# Voto de maioria numa janela de frames evita que o rotulo pisque quando a
# pessoa cruza sombra ou vira de lado.
STABILIZE_WINDOW = 6
STABILIZE_RATIO = 0.40

TRACK_IOU = 0.20
TRACK_MAX_MISSING = 18  # frames de tolerancia antes de esquecer a pessoa

# ── interface ─────────────────────────────────────────────────────────────────
SYSTEM_TITLE = "SISTEMA DE DETECCAO DE EPI"
SIDEBAR_WIDTH = 235

# Caixas tracejadas mostrando onde a IA mede cada EPI. Bom para demonstrar
# o funcionamento; desligue para uma tela de operacao mais limpa.
SHOW_ANALYSIS_ZONES = True

# ── servidor web ──────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = int(os.getenv("EPI_PORT", "5000"))
STREAM_FPS = 12
JPEG_QUALITY = 70

# ── aceleracao ────────────────────────────────────────────────────────────────
# OpenVINO acelera em iGPU Intel. Requer exportar o modelo antes:
#   yolo export model=yolov8n-pose.pt format=openvino
USE_OPENVINO = os.getenv("EPI_USE_OPENVINO", "0") == "1"
TORCH_THREADS = int(os.getenv("EPI_TORCH_THREADS", "0")) or None


# ── sobrescrita local ─────────────────────────────────────────────────────────
_OVERRIDABLE = {
    "SOURCE", "FRAME_WIDTH", "FRAME_HEIGHT", "TARGET_FPS",
    "INPUT_SIZE", "INPUT_SIZE_NEAR", "NEAR_SCENE_RATIO",
    "CONF_THRESH", "MAX_ASPECT", "MIN_KEYPOINTS", "KEYPOINT_MIN_CONF",
    "VEST_THRESH", "HELMET_THRESH",
    "EPI_METHOD", "PPE_MODEL_PATH", "PPE_CONF", "PPE_IMGSZ", "PPE_MIN_OVERLAP",
    "CLIP_MODEL", "CLIP_PRETRAINED",
    "CLIP_MIN_HEAD_PX", "CLIP_MIN_TORSO_PX", "CLIP_MIN_MARGIN",
    "STABILIZE_WINDOW", "STABILIZE_RATIO", "TRACK_IOU", "TRACK_MAX_MISSING",
    "SYSTEM_TITLE", "SIDEBAR_WIDTH", "SHOW_ANALYSIS_ZONES",
    "HOST", "PORT", "STREAM_FPS", "JPEG_QUALITY",
    "USE_OPENVINO", "TORCH_THREADS", "MODEL_PATH",
}

LOCAL_SETTINGS = ROOT / "settings_local.json"


def _apply_local_overrides() -> list[str]:
    """Aplica settings_local.json sobre os valores acima.

    Mantem a configuracao especifica da maquina fora do repositorio,
    de modo que atualizar o codigo nunca sobrescreva o ajuste local.
    """
    if not LOCAL_SETTINGS.exists():
        return []
    try:
        data = json.loads(LOCAL_SETTINGS.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    applied = []
    for key, value in data.items():
        if key in _OVERRIDABLE:
            globals()[key] = value
            applied.append(key)
    return applied


APPLIED_OVERRIDES = _apply_local_overrides()
