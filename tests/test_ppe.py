"""
Testes da integracao com o modelo treinado de EPI.

Nao roda o modelo: testa o que vem depois dele. Os nomes de classe
variam entre datasets e o Roboflow reordena os indices a cada versao,
entao a normalizacao por nome precisa aceitar as variantes comuns. E a
atribuicao precisa creditar cada capacete e colete a pessoa certa,
mesmo com gente parcialmente sobreposta no quadro.
"""

from core.detector import Detection, assign_ppe, helmet_roi, ppe_label, vest_roi
from tests.test_geometry import BOX, POSE

W, H = 640, 480


def make_person(dx: int = 0) -> Detection:
    box = (BOX[0] + dx, BOX[1], BOX[2] + dx, BOX[3])
    kps = [(x + dx, y, c) for x, y, c in POSE]
    return Detection(box=box, keypoints=kps,
                     vest_roi=vest_roi(box, kps, W, H),
                     helmet_roi=helmet_roi(box, kps, W, H))


def test_ppe_label_aceita_variantes_de_dataset():
    assert ppe_label("Hardhat") == "helmet"
    assert ppe_label("NO-Hardhat") == "no_helmet"
    assert ppe_label("Safety Vest") == "vest"
    assert ppe_label("NO-Safety Vest") == "no_vest"
    assert ppe_label("helmet") == "helmet"
    assert ppe_label("no_helmet") == "no_helmet"
    assert ppe_label("head") == "no_helmet"
    assert ppe_label("Person") is None
    assert ppe_label("Safety Cone") is None


def test_capacete_e_colete_na_pessoa_certa():
    person = make_person()
    ppe = [("helmet", (125, 45, 155, 70), 0.8),
           ("vest", (118, 100, 162, 180), 0.7)]
    assign_ppe([person], ppe, W, H)
    assert person.has_helmet and person.has_vest
    assert person.helmet_score == 0.8


def test_sem_deteccao_e_violacao():
    person = make_person()
    assign_ppe([person], [], W, H)
    assert not person.has_helmet and not person.has_vest


def test_negativo_mais_confiante_vence():
    person = make_person()
    ppe = [("helmet", (125, 45, 155, 70), 0.40),
           ("no_helmet", (125, 45, 155, 70), 0.75)]
    assign_ppe([person], ppe, W, H)
    assert not person.has_helmet


def test_colete_na_cabeca_nao_conta():
    """Caixa de colete fora do tronco nao e creditada."""
    person = make_person()
    assign_ppe([person], [("vest", (125, 40, 155, 70), 0.9)], W, H)
    assert not person.has_vest


def test_capacete_vai_para_a_pessoa_mais_proxima():
    left, right = make_person(), make_person(dx=200)
    ppe = [("helmet", (325, 45, 355, 70), 0.8)]
    assign_ppe([left, right], ppe, W, H)
    assert right.has_helmet
    assert not left.has_helmet
