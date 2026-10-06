"""
Classificacao de EPI com CLIP sobre os recortes da pose.

O detector treinado so conhece o que viu no treino: bone de uma pessoa,
num quarto, virava capacete assim que a luz ou o angulo mudavam. O CLIP
foi pre-treinado em centenas de milhoes de pares imagem-texto, entao ja
distingue "capacete de obra" de "bone" sem treino nenhum, em qualquer
ambiente.

A pose continua localizando cabeca e tronco; o CLIP so responde o que ha
em cada recorte, comparando a imagem com descricoes em texto:

  cabeca  capacete | bone/chapeu/touca | nada  -> so "capacete" conta
  tronco  colete refletivo | roupa comum
"""

from __future__ import annotations

import cv2
import numpy as np

import config
from core.tracker import iou

# Varias descricoes por classe: a media dos embeddings e mais estavel que
# uma frase so. O CLIP foi treinado em ingles, entao os textos ficam em
# ingles.
HEAD_PROMPTS = {
    "capacete": [
        "a photo of a person wearing a construction hard hat",
        "a photo of a worker wearing a plastic safety helmet",
        "a photo of a head with a hard hat",
    ],
    "bone": [
        "a photo of a person wearing a baseball cap",
        "a photo of a person wearing a cap",
        "a photo of a person wearing a beanie",
        "a photo of a person wearing a hat",
    ],
    "nada": [
        "a photo of a person with a bare head",
        "a photo of a person's hair",
        "a photo of a bald head",
    ],
}

TORSO_PROMPTS = {
    "colete": [
        "a photo of a person wearing a high-visibility safety vest",
        "a photo of a person wearing a reflective vest",
        "a photo of a worker in a hi-vis vest",
    ],
    "roupa": [
        "a photo of a person wearing a t-shirt",
        "a photo of a person wearing a shirt",
        "a photo of a person wearing a jacket",
        "a photo of a person wearing an orange t-shirt",
    ],
}

# Recortes menores que isso nao tem detalhe para o CLIP decidir.
MIN_CROP = 12


class ClipEPI:
    """Classifica recortes de cabeca e tronco por similaridade com texto."""

    def __init__(self):
        import open_clip
        import torch

        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED)
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL)
        self.head = self._encode_classes(HEAD_PROMPTS, tokenizer)
        self.torso = self._encode_classes(TORSO_PROMPTS, tokenizer)
        self._cache: list[dict] = []
        self._frame = 0

    def _encode_classes(self, prompts: dict[str, list[str]], tokenizer):
        """Um embedding por classe: media das descricoes, normalizada."""
        names, vectors = [], []
        with self._torch.no_grad():
            for name, texts in prompts.items():
                emb = self.model.encode_text(tokenizer(texts))
                emb = emb / emb.norm(dim=-1, keepdim=True)
                mean = emb.mean(dim=0)
                vectors.append(mean / mean.norm())
                names.append(name)
        return names, self._torch.stack(vectors)

    def _embed(self, frame, boxes) -> list:
        """Embedding de imagem de cada caixa, numa rodada so do modelo.

        Cabeca e tronco entram juntos: uma chamada com varios recortes
        custa bem menos que varias chamadas com um recorte cada.
        """
        from PIL import Image

        crops, slots = [], []
        for i, box in enumerate(boxes):
            if box is None:
                continue
            x1, y1, x2, y2 = box
            if x2 - x1 < MIN_CROP or y2 - y1 < MIN_CROP:
                continue
            rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            crops.append(self.preprocess(Image.fromarray(rgb)))
            slots.append(i)

        out = [None] * len(boxes)
        if crops:
            with self._torch.no_grad():
                emb = self.model.encode_image(self._torch.stack(crops))
                emb = emb / emb.norm(dim=-1, keepdim=True)
            for slot, vector in zip(slots, emb, strict=True):
                out[slot] = vector
        return out

    @staticmethod
    def _probs(vector, classes) -> dict[str, float] | None:
        if vector is None:
            return None
        names, text = classes
        p = (100.0 * vector @ text.T).softmax(dim=-1).numpy()
        return dict(zip(names, (float(v) for v in p), strict=True))

    def _classify(self, frame, boxes, classes) -> list[dict[str, float] | None]:
        """Probabilidade de cada classe para cada caixa (None se pequena)."""
        return [self._probs(v, classes) for v in self._embed(frame, boxes)]

    def assign(self, frame: np.ndarray, dets, head_boxes) -> None:
        """Preenche has_helmet/has_vest e os scores de cada deteccao.

        O veredito fica None (nao avaliado) quando o recorte e pequeno
        demais ou quando a classe vencedora ganha por margem estreita. Em
        teste, os erros se concentravam justamente em cabecas distantes,
        onde o CLIP tem poucos pixels para decidir.

        O CLIP custa mais que a pose, e ninguem poe ou tira capacete em
        uma fracao de segundo. Entao cada pessoa e reclassificada a cada
        CLIP_EVERY quadros; nos intervalos, o veredito anterior segue a
        pessoa (casada pela sobreposicao de caixa). Pessoa nova na cena e
        classificada na hora.
        """
        self._frame += 1
        pending, cache = [], []
        for i, det in enumerate(dets):
            hit = self._cached(det.box)
            if hit is not None:
                det.has_helmet, det.has_vest, det.helmet_score, det.vest_score = hit["verdict"]
                cache.append({**hit, "box": det.box})
            else:
                pending.append(i)

        if pending:
            boxes = ([head_boxes[i] for i in pending]
                     + [dets[i].vest_roi for i in pending])
            vectors = self._embed(frame, boxes)
            n = len(pending)
            for k, i in enumerate(pending):
                det = dets[i]
                self._decide(det, head_boxes[i], self._probs(vectors[k], self.head),
                             det.vest_roi, self._probs(vectors[n + k], self.torso))
                cache.append({"box": det.box, "frame": self._frame,
                              "verdict": (det.has_helmet, det.has_vest,
                                          det.helmet_score, det.vest_score)})
        self._cache = cache

    def _cached(self, box):
        """Veredito recente de uma pessoa nesta posicao, se houver."""
        best, best_iou = None, 0.5
        for entry in self._cache:
            if self._frame - entry["frame"] >= config.CLIP_EVERY:
                continue
            score = iou(box, entry["box"])
            if score >= best_iou:
                best, best_iou = entry, score
        return best

    @staticmethod
    def _decide(det, hbox, head, tbox, torso) -> None:
        det.has_helmet = det.has_vest = None
        if head and _height(hbox) >= config.CLIP_MIN_HEAD_PX:
            det.helmet_score = head["capacete"]
            other = max(v for k, v in head.items() if k != "capacete")
            if abs(det.helmet_score - other) >= config.CLIP_MIN_MARGIN:
                det.has_helmet = det.helmet_score > other
        if torso and _height(tbox) >= config.CLIP_MIN_TORSO_PX:
            det.vest_score = torso["colete"]
            if abs(det.vest_score - torso["roupa"]) >= config.CLIP_MIN_MARGIN:
                det.has_vest = det.vest_score > torso["roupa"]


def _height(box) -> int:
    return box[3] - box[1] if box else 0
