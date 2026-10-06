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

    def _classify(self, frame, boxes, classes) -> list[dict[str, float] | None]:
        """Probabilidade de cada classe para cada caixa (None se pequena)."""
        from PIL import Image

        names, text = classes
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

        out: list[dict[str, float] | None] = [None] * len(boxes)
        if not crops:
            return out
        with self._torch.no_grad():
            emb = self.model.encode_image(self._torch.stack(crops))
            emb = emb / emb.norm(dim=-1, keepdim=True)
            probs = (100.0 * emb @ text.T).softmax(dim=-1).numpy()
        for slot, p in zip(slots, probs, strict=True):
            out[slot] = dict(zip(names, (float(v) for v in p), strict=True))
        return out

    def assign(self, frame: np.ndarray, dets, head_boxes) -> None:
        """Preenche has_helmet/has_vest e os scores de cada deteccao."""
        heads = self._classify(frame, head_boxes, self.head)
        torsos = self._classify(frame, [d.vest_roi for d in dets], self.torso)
        for det, head, torso in zip(dets, heads, torsos, strict=True):
            if head:
                det.helmet_score = head["capacete"]
                det.has_helmet = max(head, key=head.get) == "capacete"
            if torso:
                det.vest_score = torso["colete"]
                det.has_vest = det.vest_score > torso["roupa"]
