"""
Testes da avaliacao em imagens rotuladas (treino/avaliar.py).

O que importa aqui e a ligacao entre o rotulo e a pessoa detectada: se
uma caixa de cabeca for casada com a pessoa errada, o acerto medido nao
diz nada sobre o detector.
"""

import sys
from pathlib import Path

from tests.test_ppe import H, W, make_person

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "treino"))
import avaliar  # noqa: E402

NAMES = ["Hardhat", "NO-Hardhat", "NO-Safety Vest", "Safety Vest"]


def test_cabeca_rotulada_vai_para_a_pessoa_certa():
    left, right = make_person(), make_person(dx=200)
    head = (325, 45, 355, 75)  # cabeca da pessoa da direita
    assert avaliar.match("no_helmet", head, [left, right], W, H) is right


def test_tronco_rotulado_vai_para_a_pessoa_certa():
    left, right = make_person(), make_person(dx=200)
    torso = (118, 100, 162, 180)  # tronco da pessoa da esquerda
    assert avaliar.match("no_vest", torso, [left, right], W, H) is left


def test_rotulo_longe_de_todos_fica_sem_pessoa():
    assert avaliar.match("helmet", (560, 400, 600, 440), [make_person()], W, H) is None


def test_le_caixa_e_poligono(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n"
                     "2 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n")
    boxes = avaliar.read_labels(label, NAMES, 100, 100)
    assert boxes[0] == ("helmet", (40, 40, 60, 60))
    assert boxes[1] == ("no_vest", (10, 10, 30, 40))
