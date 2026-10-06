#!/usr/bin/env python3
"""
Gera rotulos de treino a partir de um video em que o EPI nao muda.

Num video gravado para treino (todo de bone, todo sem colete...) a classe
e a mesma em todos os quadros; o trabalho manual seria so desenhar as
caixas. A pose ja localiza cabeca e tronco, entao este script extrai os
quadros e escreve as caixas no formato YOLO, pronto para subir no Roboflow.

Uso:

    python treino/autorrotular.py bone_azul.mp4 --cabeca sem-capacete --tronco sem-colete
    python treino/autorrotular.py com_colete.mp4 --cabeca sem-capacete --tronco colete --fps 2

Saida (por padrao em rotulos/<nome do video>/):

    images/    quadros extraidos
    labels/    uma caixa de cabeca e uma de tronco por pessoa
    data.yaml  nomes das classes, iguais aos do dataset no Roboflow
    conferir/  os mesmos quadros com as caixas desenhadas, para revisar

Quadros em que a pose nao e confiavel (ombros fora do quadro, numero de
pessoas diferente do esperado) sao descartados em vez de rotulados pela
metade: um tronco sem rotulo ensina ao modelo que tronco sem colete e
fundo, que foi o que derrubou o NO-Safety Vest no treino v3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.detector import (  # noqa: E402
    EYE_L,
    EYE_R,
    HIP_L,
    HIP_R,
    NOSE,
    SHOULDER_L,
    SHOULDER_R,
    _overlap,
    head_geometry,
    head_region,
    keypoint,
    ppe_label,
)

# Mesma ordem e grafia do dataset no Roboflow; o Roboflow casa pelo nome.
CLASSES = ["Hardhat", "NO-Hardhat", "NO-Safety Vest", "Safety Vest"]
CABECA = {"capacete": "Hardhat", "sem-capacete": "NO-Hardhat"}
TRONCO = {"colete": "Safety Vest", "sem-colete": "NO-Safety Vest"}

# Ombros abaixo disso deslocam as caixas; melhor descartar o quadro.
SHOULDER_MIN_CONF = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai quadros de um video e gera rotulos YOLO de cabeca e tronco")
    parser.add_argument("video", help="arquivo de video")
    parser.add_argument("--cabeca", required=True, choices=sorted(CABECA),
                        help="o que a pessoa tem na cabeca durante todo o video")
    parser.add_argument("--tronco", required=True, choices=sorted(TRONCO),
                        help="se a pessoa esta de colete durante todo o video")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="quadros extraidos por segundo (padrao: 1)")
    parser.add_argument("--pessoas", type=int, default=1,
                        help="pessoas esperadas no quadro; quadros com outra "
                             "quantidade sao descartados (padrao: 1)")
    parser.add_argument("--conf-min", type=float, default=0.5,
                        help="ignora deteccoes de pessoa abaixo desta confianca "
                             "antes de contar (padrao: 0.5)")
    parser.add_argument("--modelo-caixas", default=config.PPE_MODEL_PATH,
                        help="modelo de EPI usado so para desenhar as caixas de "
                             "cabeca (padrao: models/epi.pt; sem ele, usa a pose)")
    parser.add_argument("--saida", default=None,
                        help="pasta de saida (padrao: rotulos/<nome do video>)")
    return parser.parse_args()


def head_box(det, frame_w: int, frame_h: int, model_boxes):
    """Caixa da cabeca, do topo do cranio ao queixo. Devolve (caixa, fonte).

    Prefere a caixa do modelo de EPI ja treinado: ele aprendeu a convencao
    de caixa de cabeca do dataset. So a geometria da caixa e aproveitada;
    a classe vem do argumento --cabeca, entao confundir bone com capacete
    nao importa aqui.

    Sem caixa do modelo, cai para a pose. head_geometry e folgada de
    proposito (procura capacete com margem) e, com a pessoa perto da
    camera, estimava a cabeca descendo ate o peito; por isso a base e
    limitada a linha dos ombros e a largura a proporcao de uma cabeca.
    """
    cx, half_w, top, chin = head_geometry(det.box, det.keypoints)
    shoulder_y = min(keypoint(det.keypoints, SHOULDER_L)[1],
                     keypoint(det.keypoints, SHOULDER_R)[1])
    span = max(shoulder_y - top, 8.0)
    nose = keypoint(det.keypoints, NOSE, 0.5)
    if nose and nose[1] > top:
        chin = top + (nose[1] - top) / 0.62  # nariz fica a ~62% da altura
    bottom = min(max(chin, top + span * 0.5), shoulder_y - span * 0.08)

    region = head_region(det.box, det.keypoints, frame_w, frame_h)
    best = None
    for box, conf in model_boxes:
        score = _overlap(box, region)
        if score >= 0.6 and (best is None or conf > best[1]):
            best = (box, conf)
    if best:
        # Para bone o modelo v1 ve "capacete" e desenha a caixa so em volta
        # da aba; NO-Hardhat no dataset cobre a cabeca inteira, entao a base
        # desce ate o queixo estimado. O inverso tambem acontece: de perto,
        # o modelo as vezes marca so o rosto e deixa o bone de fora, entao o
        # topo sobe ate o topo da pessoa (que e o topo do bone).
        x1, y1, x2, y2 = best[0]
        y1 = min(y1, int(top))
        return (x1, y1, x2, min(frame_h - 1, max(y2, int(bottom)))), "modelo"

    half_w = min(half_w * 0.70, (bottom - top) * 0.42)
    return ((max(0, int(cx - half_w)), max(0, int(top)),
             min(frame_w - 1, int(cx + half_w)), min(frame_h - 1, int(bottom))),
            "pose")


def head_cut_off(det, frame_h: int) -> bool:
    """Cabeca saindo pelo topo do quadro, sem rosto visivel.

    Acontece na webcam quando a pessoa levanta: a caixa da pessoa encosta
    no topo e a "cabeca" estimada cai no pescoco. Rosto visivel indica que
    ao menos a cabeca esta no quadro, e esse caso continua aproveitado.
    """
    if det.box[1] > frame_h * 0.02:
        return False
    return not any(keypoint(det.keypoints, i, 0.5) for i in (NOSE, EYE_L, EYE_R))


def torso_box(det):
    """Tronco visivel, ou None se quase nao aparece.

    Com a bacia fora do quadro, vest_roi estima a base por percentual da
    caixa e, com a pessoa perto da camera, gerava uma faixa fina na borda.
    Aqui a base vai ate onde a pessoa sai do quadro; se sobra menos tronco
    que meia largura de ombros, nao ha o que rotular.
    """
    x1, top, x2, bottom = det.vest_roi
    hips = [keypoint(det.keypoints, i) for i in (HIP_L, HIP_R)]
    if not any(hips):
        bottom = det.box[3]
    sl = keypoint(det.keypoints, SHOULDER_L)
    sr = keypoint(det.keypoints, SHOULDER_R)
    shoulder_w = abs(sr[0] - sl[0])
    if bottom - min(sl[1], sr[1]) < max(shoulder_w * 0.5, 12):
        return None
    return (x1, top, x2, bottom)


def yolo_line(cls: str, box, frame_w: int, frame_h: int) -> str:
    x1, y1, x2, y2 = box
    return (f"{CLASSES.index(cls)} {(x1 + x2) / 2 / frame_w:.6f} "
            f"{(y1 + y2) / 2 / frame_h:.6f} {(x2 - x1) / frame_w:.6f} "
            f"{(y2 - y1) / frame_h:.6f}")


def main() -> int:
    args = parse_args()
    video = Path(args.video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"Nao foi possivel abrir: {video}", file=sys.stderr)
        return 1

    out = Path(args.saida) if args.saida else Path("rotulos") / video.stem
    for sub in ("images", "labels", "conferir"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # so a pose interessa aqui; o metodo de cor evita carregar o modelo de EPI
    config.EPI_METHOD = "cor"
    from core.detector import EPIDetector
    detector = EPIDetector(imgsz=config.INPUT_SIZE_NEAR)
    box_model = None
    if Path(args.modelo_caixas).exists():
        from ultralytics import YOLO
        box_model = YOLO(args.modelo_caixas)
        print(f"Caixas de cabeca: modelo {args.modelo_caixas}, com a pose de reserva")
    else:
        print(f"Caixas de cabeca: so a pose ({args.modelo_caixas} nao encontrado)")

    head_cls, torso_cls = CABECA[args.cabeca], TRONCO[args.tronco]
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(video_fps / args.fps))

    index = saved = 0
    skipped = {"pessoas": 0, "ombros": 0, "cabeca": 0}
    sources = {"modelo": 0, "pose": 0}
    torso_missing = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        if (index - 1) % step:
            continue

        h, w = frame.shape[:2]
        # Em ambiente domestico o detector acha "pessoas" em roupa pendurada
        # e mochila, com confianca 0.2-0.3; a pessoa real fica acima de 0.8.
        # Sem esse corte metade dos quadros era descartada por contagem.
        dets = [d for d in detector.detect(frame)
                if d.confidence >= args.conf_min]
        if len(dets) != args.pessoas:
            skipped["pessoas"] += 1
            continue
        if any(keypoint(d.keypoints, i, SHOULDER_MIN_CONF) is None
               for d in dets for i in (SHOULDER_L, SHOULDER_R)):
            skipped["ombros"] += 1
            continue
        if any(head_cut_off(d, h) for d in dets):
            skipped["cabeca"] += 1
            continue

        model_boxes = []
        if box_model is not None:
            result = box_model(frame, conf=0.25, imgsz=config.PPE_IMGSZ,
                               verbose=False)[0]
            for cls, box, conf in zip(result.boxes.cls.tolist(),
                                      result.boxes.xyxy.tolist(),
                                      result.boxes.conf.tolist(), strict=True):
                if ppe_label(result.names[int(cls)]) in ("helmet", "no_helmet"):
                    model_boxes.append((tuple(int(v) for v in box), conf))

        name = f"{video.stem}_{index:05d}"
        lines, preview = [], frame.copy()
        for det in dets:
            head, source = head_box(det, w, h, model_boxes)
            sources[source] += 1
            boxes = [(head_cls, head, (0, 220, 255))]
            torso = torso_box(det)
            if torso:
                boxes.append((torso_cls, torso, (0, 200, 0)))
            else:
                torso_missing += 1
            for cls, box, color in boxes:
                lines.append(yolo_line(cls, box, w, h))
                cv2.rectangle(preview, box[:2], box[2:], color, 2)
                cv2.putText(preview, cls, (box[0], max(14, box[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        cv2.imwrite(str(out / "images" / f"{name}.jpg"), frame)
        cv2.imwrite(str(out / "conferir" / f"{name}.jpg"), preview)
        (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
        saved += 1

    cap.release()
    (out / "data.yaml").write_text(
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n", encoding="utf-8")

    print(f"Rotulados : {saved} quadros em {out}")
    print(f"Descartados: {skipped['pessoas']} (numero de pessoas diferente de "
          f"{args.pessoas}), {skipped['ombros']} (ombros fora do quadro ou incertos), "
          f"{skipped['cabeca']} (cabeca cortada no topo)")
    print(f"Cabecas    : {sources['modelo']} pelo modelo, {sources['pose']} pela pose | "
          f"tronco omitido em {torso_missing} (quase fora do quadro)")
    print(f"Revise as caixas em {out / 'conferir'} antes de subir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
