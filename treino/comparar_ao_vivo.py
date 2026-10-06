#!/usr/bin/env python3
"""
Compara o metodo de cor e modelos de EPI nos mesmos quadros da webcam.

A tela guia o teste em fases (sem bone, depois com bone). Em cada quadro
com exatamente uma pessoa, todos os metodos julgam a mesma deteccao de
pose, entao a diferenca entre eles e so a decisao de EPI. Nenhum quadro e
salvo; ao final sai uma tabela com quantas vezes cada metodo acertou.

Uso:

    python treino/comparar_ao_vivo.py models/epi.pt models/epi_v4.pt
    python treino/comparar_ao_vivo.py models/epi.pt models/epi_v4.pt --segundos 30

Q ou ESC encerra antes do fim (a tabela sai com o que foi medido).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.detector import assign_ppe, ppe_label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara metodos de EPI ao vivo na webcam")
    parser.add_argument("modelos", nargs="+", help="arquivos .pt a comparar")
    parser.add_argument("--source", type=int, default=0, help="indice da webcam")
    parser.add_argument("--segundos", type=int, default=20,
                        help="duracao de cada fase de medicao (padrao: 20)")
    return parser.parse_args()


def detect_ppe(model, frame):
    result = model(frame, conf=config.PPE_CONF, imgsz=config.PPE_IMGSZ,
                   verbose=False)[0]
    found = []
    for cls, box, conf in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist(),
                              result.boxes.conf.tolist(), strict=True):
        label = ppe_label(result.names[int(cls)])
        if label:
            found.append((label, tuple(int(v) for v in box), float(conf)))
    return found


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    # a pose e a mesma para todos; o metodo de cor vem junto de graca
    config.EPI_METHOD = "cor"
    from core.detector import EPIDetector
    detector = EPIDetector(imgsz=640)
    models = {Path(p).stem: YOLO(p) for p in args.modelos}
    names = ["cor", *models]

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Nao foi possivel abrir a webcam", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # (texto na tela, duracao, chave da medicao ou None para transicao)
    phases = [("PREPARE-SE: SEM BONE", 6, None),
              ("SEM BONE - ande, vire de lado", args.segundos, "sem bone"),
              ("COLOQUE O BONE", 8, None),
              ("COM BONE - ande, vire de lado", args.segundos, "com bone")]
    stats = {key: {"quadros": 0, **{n: 0 for n in names}}
             for _, _, key in phases if key}
    vest_fp = dict.fromkeys(names, 0)

    window = "Comparacao de EPI  |  Q para sair"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    started = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        elapsed = time.time() - started
        t, phase = 0.0, None
        for phase in phases:
            if elapsed < t + phase[1]:
                break
            t += phase[1]
        else:
            break
        text, duration, key = phase
        remaining = int(t + duration - elapsed) + 1

        h, w = frame.shape[:2]
        dets = [d for d in detector.detect(frame) if d.confidence >= 0.5]
        verdicts = {}
        if len(dets) == 1:
            det = dets[0]
            verdicts["cor"] = (det.has_helmet, det.has_vest)
            for name, model in models.items():
                copy = dataclasses.replace(det)
                assign_ppe([copy], detect_ppe(model, frame), w, h)
                verdicts[name] = (copy.has_helmet, copy.has_vest)
            if key:
                stats[key]["quadros"] += 1
                for name, (helmet, vest) in verdicts.items():
                    stats[key][name] += not helmet  # certo: ninguem usa capacete
                    vest_fp[name] += vest           # ninguem usa colete

        view = frame.copy()
        cv2.rectangle(view, (0, 0), (w, 46), (0, 0, 0), -1)
        cv2.putText(view, f"{text}  {remaining}s", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        y = 80
        for name in names:
            if name in verdicts:
                helmet = verdicts[name][0]
                label = "CAPACETE" if helmet else "sem capacete"
                color = (0, 0, 255) if helmet else (0, 200, 0)
            else:
                label, color = "(nenhuma pessoa)", (200, 200, 200)
            cv2.putText(view, f"{name:>8}: {label}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            y += 30
        cv2.imshow(window, view)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nAcerto no capacete (ninguem estava de capacete):\n")
    print(f"{'metodo':>10} | " + " | ".join(f"{k:>16}" for k in stats)
          + " | colete falso")
    total = sum(s["quadros"] for s in stats.values())
    for name in names:
        cells = []
        for s in stats.values():
            n = s["quadros"]
            cells.append(f"{s[name] / n * 100:5.0f}% ({s[name]:>3}/{n:<3})"
                         if n else f"{'-':>16}")
        fp = f"{vest_fp[name] / total * 100:5.0f}%" if total else "-"
        print(f"{name:>10} | " + " | ".join(cells) + f" | {fp:>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
