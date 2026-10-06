#!/usr/bin/env python3
"""
Mede o acerto do detector de EPI numa pasta de imagens rotuladas.

Aceita pastas no formato YOLO (images/, labels/ e data.yaml), que e o que
o Roboflow exporta e o que treino/autorrotular.py gera. Cada caixa de
cabeca (Hardhat / NO-Hardhat) e de colete (Safety Vest / NO-Safety Vest)
e casada com a pessoa que o sistema detectou naquela regiao do corpo, e o
veredito do sistema e comparado com o rotulo.

Uso:

    python treino/avaliar.py teste/obra
    python treino/avaliar.py rotulos/boneazul rotulos/sembonew --metodo clip
    python treino/avaliar.py teste/obra --erros saida_erros

Os erros sao separados pelo peso que tem num sistema de seguranca:

  violacao escondida  pessoa sem EPI, sistema diz que esta com (grave)
  alarme falso        pessoa com EPI, sistema diz que esta sem
  em duvida           o sistema respondeu "nao avaliado" (pessoa pequena
                      demais ou classificador indeciso); nao e erro
  nao encontrada      a pose nao encontrou a pessoa do rotulo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.detector import _overlap, head_region, ppe_label  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Fracao da caixa rotulada que precisa cair na regiao da pessoa detectada.
MATCH_HEAD = 0.5
MATCH_TORSO = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mede o acerto de capacete e colete em imagens rotuladas")
    parser.add_argument("pastas", nargs="+",
                        help="pastas no formato YOLO (images/, labels/, data.yaml)")
    parser.add_argument("--metodo", choices=("auto", "clip", "modelo", "cor"),
                        default=None, help="metodo de EPI (padrao: o do config)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="resolucao da pose (padrao: 640)")
    parser.add_argument("--erros", default=None,
                        help="pasta onde salvar as imagens com erro, para revisar")
    return parser.parse_args()


def read_names(folder: Path) -> list[str]:
    """Nomes das classes do data.yaml, na ordem dos indices."""
    import yaml

    for candidate in (folder / "data.yaml", folder.parent / "data.yaml"):
        if candidate.exists():
            data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
            names = data["names"]
            return list(names.values()) if isinstance(names, dict) else list(names)
    raise FileNotFoundError(f"data.yaml nao encontrado em {folder}")


def find_split(folder: Path) -> list[tuple[Path, Path]]:
    """Pares (imagem, rotulo). Aceita a pasta raiz ou um split do Roboflow."""
    roots = [folder] if (folder / "images").is_dir() else [
        folder / s for s in ("test", "valid", "train") if (folder / s / "images").is_dir()]
    if folder.name not in ("test", "valid", "train") and (folder / "test").is_dir():
        roots = [folder / "test"]  # export do Roboflow: avalia so o teste
    pairs = []
    for root in roots:
        for img in sorted((root / "images").iterdir()):
            if img.suffix.lower() in IMAGE_EXT:
                pairs.append((img, root / "labels" / f"{img.stem}.txt"))
    return pairs


def read_labels(path: Path, names: list[str], w: int, h: int):
    """Caixas rotuladas como (rotulo normalizado, (x1, y1, x2, y2))."""
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        label = ppe_label(names[int(parts[0])])
        if not label:
            continue
        if len(parts) == 5:
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        else:  # poligono: a caixa e o envelope dos pontos
            xs = [float(v) for v in parts[1::2]]
            ys = [float(v) for v in parts[2::2]]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        boxes.append((label, (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))))
    return boxes


def match(label: str, box, dets, frame_w: int, frame_h: int):
    """Pessoa detectada a que a caixa rotulada pertence, ou None."""
    head = label in ("helmet", "no_helmet")
    best, best_score = None, MATCH_HEAD if head else MATCH_TORSO
    for det in dets:
        # o tronco e casado com a pessoa inteira: a faixa de analise do
        # colete e estreita e deixava troncos rotulados sem dono
        region = (head_region(det.box, det.keypoints, frame_w, frame_h)
                  if head else det.box)
        score = _overlap(box, region)
        if score >= best_score:
            best, best_score = det, score
    return best


def new_counts() -> dict[str, int]:
    return {"total": 0, "certo": 0, "escondida": 0, "alarme": 0,
            "duvida": 0, "nao_encontrada": 0}


def main() -> int:
    args = parse_args()
    if args.metodo:
        config.EPI_METHOD = args.metodo
    # imagens independentes: o veredito de uma nao pode ser reaproveitado
    # na seguinte so porque a pessoa esta na mesma posicao
    config.CLIP_EVERY = 1
    from core.detector import EPIDetector
    detector = EPIDetector(imgsz=args.imgsz)
    print(f"Metodo de EPI: {detector.method}\n")

    errors_dir = Path(args.erros) if args.erros else None
    if errors_dir:
        errors_dir.mkdir(parents=True, exist_ok=True)

    overall = {"capacete": new_counts(), "colete": new_counts()}
    for folder in map(Path, args.pastas):
        names = read_names(folder)
        counts = {"capacete": new_counts(), "colete": new_counts()}
        for img_path, label_path in find_split(folder):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            truth = read_labels(label_path, names, w, h)
            if not truth:
                continue
            dets = detector.detect(frame)
            wrong = False
            for label, box in truth:
                kind = "capacete" if label in ("helmet", "no_helmet") else "colete"
                c = counts[kind]
                c["total"] += 1
                det = match(label, box, dets, w, h)
                if det is None:
                    c["nao_encontrada"] += 1
                    continue
                expected = label in ("helmet", "vest")
                said = det.has_helmet if kind == "capacete" else det.has_vest
                if said is None:
                    c["duvida"] += 1
                elif said == expected:
                    c["certo"] += 1
                else:
                    c["escondida" if said else "alarme"] += 1
                    wrong = True
                    if errors_dir:
                        cv2.rectangle(frame, box[:2], box[2:], (0, 0, 255), 2)
                        cv2.putText(frame, f"{kind}: sistema={'com' if said else 'sem'}",
                                    (box[0], max(14, box[1] - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if wrong and errors_dir:
                cv2.imwrite(str(errors_dir / f"{folder.name}_{img_path.name}"), frame)

        report(folder.name, counts)
        for kind in counts:
            for key, value in counts[kind].items():
                overall[kind][key] += value

    if len(args.pastas) > 1:
        report("TOTAL", overall)
    return 0


def report(title: str, counts) -> None:
    print(f"== {title}")
    for kind, c in counts.items():
        if not c["total"]:
            continue
        judged = c["total"] - c["nao_encontrada"] - c["duvida"]
        rate = f"{c['certo'] / judged * 100:5.1f}%" if judged else "    -"
        print(f"  {kind:8s} acerto {rate} ({c['certo']}/{judged}) | "
              f"violacao escondida {c['escondida']} | alarme falso {c['alarme']} | "
              f"em duvida {c['duvida']} | nao encontrada {c['nao_encontrada']}")
    print()


if __name__ == "__main__":
    sys.exit(main())
