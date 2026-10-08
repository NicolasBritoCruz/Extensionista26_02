# EPI Detect: guia rápido do grupo

Detector de **capacete** e **colete refletivo** em vídeo (webcam, arquivo ou
câmera IP), rodando em CPU comum. Partimos do projeto
[Pravatta/epi-detect](https://github.com/Pravatta/epi-detect) e trocamos a
forma de decidir o EPI: no lugar da análise de cor, usamos o **CLIP**, um
modelo pré-treinado que diferencia capacete, boné e cabeça descoberta.

---

## Como funciona (visão geral)

```
câmera/vídeo
   │
   ▼
YOLOv8-pose ──► acha cada pessoa e 17 pontos do corpo (ombros, quadril, rosto...)
   │
   ▼
recortes ─────► cabeça e tronco de cada pessoa, a partir dos pontos do corpo
   │
   ▼
CLIP ─────────► cabeça: capacete | boné/chapéu | nada
   │            tronco: colete   | roupa comum
   │            (pouca certeza ou pessoa muito pequena = "não avaliado")
   ▼
rastreador ───► identidade de cada pessoa entre quadros + voto de maioria
   │            (o rótulo não fica piscando)
   ▼
tela ─────────► esqueleto colorido, rótulos e painel com contadores
```

Cores na tela: **amarelo** = capacete OK, **verde** = colete OK,
**vermelho** = EPI ausente, **cinza com "?"** = não avaliado (o sistema
não tem certeza e prefere não chutar).

---

## Estrutura do projeto

```
Extensionista26_02/
├── detect.py              Detecção ao vivo numa janela (webcam, vídeo ou RTSP)
├── render.py              Processa um vídeo e salva outro, anotado
├── server.py              Interface web (Flask), com zonas de risco
├── iniciar.bat            Menu para Windows: instala tudo e abre as opções
├── config.py              Todos os parâmetros, cada um com o motivo do valor
├── auto_config.py         Detecção de hardware (utilitário, fora do fluxo principal)
├── requirements.txt       Dependências para rodar
├── requirements-dev.txt   Dependências para testes e lint
│
├── core/                  O detector em si
│   ├── detector.py        Pose, recortes de cabeça/tronco e escolha do método
│   ├── clip_epi.py        Classificação com CLIP (método padrão)
│   ├── tracker.py         Rastreamento entre quadros, voto e contagem do painel
│   ├── overlay.py         Desenho: esqueleto, rótulos, painel lateral
│   ├── zones.py           Zonas de risco da interface web
│   └── capture.py         Leitura da câmera em thread (usado pelo server.py)
│
├── treino/                Ferramentas de dados e avaliação
│   ├── avaliar.py         Mede o acerto em imagens marcadas (formato YOLO)
│   ├── autorrotular.py    Marca automaticamente os quadros de um vídeo
│   ├── comparar_ao_vivo.py  Compara métodos lado a lado na webcam
│   └── treinar_epi.ipynb  Notebook do Colab para treinar um YOLO de EPI
│
├── tests/                 Testes automáticos (pytest)
├── templates/             Página da interface web
├── demo/                  Vídeos de demonstração (já vêm anotados)
└── docs/                  Documentação e imagens do projeto original
```

**Pastas que só existem no computador de cada um** (ficam fora do Git):

| Pasta | O que tem |
|---|---|
| `.venv/` | Ambiente Python, criado na instalação |
| `models/` | Pesos do YOLO de EPI treinado (opcional, ver abaixo) |
| `input/` | Vídeos próprios para marcar |
| `rotulos/` | Quadros marcados pelo `autorrotular.py` |
| `teste/` | Conjuntos de teste (ex.: o export do Roboflow) |

Essas pastas podem ter fotos de pessoas, então **não devem ir para o GitHub**.

### Métodos de decisão do EPI

Escolhido por `EPI_METHOD` em `config.py`:

| Método | Como decide | Quando é usado |
|---|---|---|
| `clip` | CLIP pré-treinado nos recortes | **Padrão.** É o mais robusto a troca de ambiente |
| `modelo` | YOLO treinado por nós (`models/epi.pt`) | Só se o arquivo existir |
| `cor` | Análise de cor do projeto original | Referência; confunde boné com capacete |
| `auto` | Usa `clip` se instalado; senão `modelo`; senão `cor` | Valor de fábrica |

---

## Como rodar

### 1. Pré-requisito

**Python 3.10 ou mais novo**, com a opção "Add Python to PATH" marcada na
instalação.

### 2. Baixar e instalar

```bash
git clone https://github.com/NicolasBritoCruz/Extensionista26_02.git
cd Extensionista26_02
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

No Linux/macOS, ative com `source .venv/bin/activate`.

> **Atalho no Windows:** dar dois cliques em `iniciar.bat` faz tudo isso
> sozinho na primeira vez e abre um menu com as opções.

### 3. Rodar

```bash
python detect.py
```

**A primeira execução demora**, porque baixa os modelos uma vez só:
YOLOv8-pose (~7 MB) e CLIP (~350 MB). Depois disso, abre em segundos. O
terminal deve mostrar `Metodo de EPI: clip`.

Teclas na janela: **Q** sai, **Z** mostra/esconde as caixas de análise,
**Espaço** pausa.

Outras fontes de vídeo:

```bash
python detect.py --source 1                    # segunda webcam
python detect.py --source caminho/video.mp4    # arquivo
python render.py caminho/video.mp4             # gera caminho/video_epi.mp4
```

### 4. Testes (para quem for mexer no código)

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

### 5. Medir o acerto (opcional)

Baixe um dataset do Roboflow em formato **YOLOv8**, descompacte em
`teste/obra/` e rode:

```bash
python treino/avaliar.py teste/obra --erros teste/erros
```

Ele mostra o acerto de capacete e colete, separando **violação escondida**
(erro grave), **alarme falso** e **em dúvida**. As imagens com erro ficam
em `teste/erros/`. Referência atual nas fotos de obra: **95% no capacete
e 93% no colete**.

---

## Ajustes sem mexer no código

Crie um arquivo `settings_local.json` na raiz com só o que quiser mudar.
Ele fica fora do Git:

```json
{
  "EPI_METHOD": "cor",
  "SOURCE": 1
}
```

---

## Problemas conhecidos

- **A interface web (`server.py`) abre, mas não mostra o percentual nem o
  registro de eventos.** A página procura campos que o servidor não envia
  mais (`with_vest`, `without_vest`). O problema já vinha do original.
- **As zonas de risco da interface web quebram ao checar uma pessoa**
  (`core/zones.py` usa `detection.x1`, que não existe; o certo é
  `detection.box`).
- **O `models/epi.pt` não vem no repositório** (pesos não são versionados).
  Ele só é necessário para o método `modelo`; o padrão (`clip`) não usa.
- **A janela fecha sozinha logo ao abrir:** a webcam provavelmente está em
  uso por outro programa, ou o Windows está bloqueando o acesso em
  *Configurações → Privacidade → Câmera*. Teste também `--source 1`.
- **Pessoas muito longe da câmera** costumam aparecer como "não avaliado":
  a cabeça tem poucos pixels para o CLIP decidir.

---

## Licença

**AGPL-3.0**, herdada do Ultralytics YOLOv8 e do projeto original. Se o
código for publicado ou oferecido como serviço, ele precisa continuar
aberto e com a mesma licença.
