"""Portuguese (pt-BR) version of the results page (``site --lang pt``).

The page is generated in English; ``translate_html`` then walks its visible
text nodes (outside ``<pre>``, ``<code>``, ``<style>`` and ``<script>``) and
replaces each one from ``EXACT`` (fixed text) or ``PATTERNS`` (text that
contains numbers or names that change from run to run), then writes decimal
numbers with a comma. Model names (UFLD v0.3 gated, ...) are kept as they are.
Hand-written material has its own Portuguese files (``*.pt.md``), which the
page reads directly. ``untranslated`` lists English text left on a page so a
new sentence in ``site.py`` is noticed (``tests/test_site.py``).
"""

from __future__ import annotations

import html
import re

EXACT: dict[str, str] = {
    # header
    "TAC-UFLD: temporal lane detection pilots": "TAC-UFLD: pilotos de detecção temporal de faixas",
    "Can a lightweight lane detector that looks at the previous frames beat the single-frame UFLD baseline? UFLD "
    "(ResNet-18) and a small CNN, each with temporal variants, trained on the ELAS ego-lane dataset and tested on "
    "three road scenes never used for training or model selection.":
        "Um detector de faixas leve que também olha os quadros anteriores consegue superar a UFLD baseline, que usa "
        "um único quadro? A UFLD (ResNet-18) e uma CNN pequena, cada uma com variantes temporais, foram treinadas na "
        "base ELAS de faixas do próprio veículo e testadas em três cenas de estrada nunca usadas no treino nem na "
        "seleção de modelos.",
    "dataset ELAS": "base ELAS",
    "indicative pilot, not the full protocol": "piloto indicativo, não o protocolo completo",
    "Authors:": "Autores:",
    # summary
    "Where the research stands": "Onde a pesquisa está",
    "Question: can a lightweight lane detector that also looks at the previous frames beat the single-frame UFLD "
    "baseline, and run on an in-car embedded system? Everything below comes from pilot runs on ELAS (2–3 seeds, "
    "short training, no hyper-parameter search), so it shows directions, not final claims.":
        "Pergunta: um detector de faixas leve que também olha os quadros anteriores consegue superar a UFLD baseline "
        "de um quadro e rodar em um sistema embarcado no carro? Tudo abaixo vem de execuções piloto no ELAS (2–3 "
        "sementes, treino curto, sem busca de hiperparâmetros); por isso indica direções, não conclusões finais.",
    "Overfitting fixed": "Overfitting corrigido",
    "UFLD baseline, held-out lane F1: original augmentation vs + geometric augmentation (3 seeds)":
        "UFLD baseline, lane F1 no teste: aumento de dados original × com aumento geométrico (3 sementes)",
    "Clean frames": "Quadros limpos",
    "Occluded current frame": "Quadro atual ocluído",
    "Kalman tracker on the output": "Filtro de Kalman na saída",
    "for every model, with unchanged lane F1: the cheap temporal reference a learned model has to beat":
        "em todos os modelos, sem mudar o lane F1: a referência temporal barata que um modelo aprendido precisa "
        "superar",
    "Lightweight model": "Modelo leve",
    "1 · Overfitting: what fixed it": "1 · Overfitting: o que resolveu",
    "UFLD baseline trained with each recipe (same split, seeds and epochs); held-out lane F1. Geometric "
    "augmentation (shift, zoom, rotation, perspective) is the fix; flips and heavy regularisation are not.":
        "UFLD baseline treinada com cada receita (mesma divisão, sementes e épocas); lane F1 no teste. O aumento "
        "geométrico (deslocamento, zoom, rotação, perspectiva) resolve; espelhamento e regularização forte não.",
    "2 · Where time helps: an occluded current frame": "2 · Onde o tempo ajuda: quadro atual ocluído",
    "Held-out lane F1 with clean frames (hollow) and with the current frame occluded by boxes while the earlier "
    "frames stay clean (filled); the number is the occluded F1 and its change. Single-frame models only see the "
    "damaged image.":
        "Lane F1 no teste com quadros limpos (vazado) e com o quadro atual coberto por retângulos enquanto os "
        "anteriores ficam limpos (cheio); o número é o F1 com oclusão e sua variação. Modelos de um quadro só veem "
        "a imagem danificada.",
    "3 · Accuracy against size": "3 · Acurácia versus tamanho",
    "(longer training).": "(treino mais longo).",
    "clean current frame": "quadro atual limpo",
    "occluded current frame": "quadro atual ocluído",
    # augmentation arms (chart and table labels)
    "geometric": "geométrico", "geometric flip": "geométrico + espelhamento",
    "photometric extended": "fotométrico estendido", "none": "nenhum", "photometric": "fotométrico",
    "regularised": "regularizado", "backbone lr ×0.1": "lr do backbone ×0,1",
    "frozen stem + layer1": "stem + layer1 congelados", "current-frame degradation": "degradação do quadro atual",
    "small head (256)": "cabeça pequena (256)", "more scenes (18)": "mais cenas (18)", "reference": "referência",
    # summary lists and choice section: the Portuguese blocks are used directly; the English ones are dropped
    # models section
    "The models": "Os modelos",
    "Two families, each a single-frame baseline and temporal variants built on it. All of them read the current "
    "frame and predict the two ego lanes of that frame; the temporal ones also read earlier frames.":
        "Duas famílias, cada uma com uma baseline de um quadro e variantes temporais construídas sobre ela. Todos "
        "leem o quadro atual e preveem as duas faixas do veículo nesse quadro; os temporais também leem quadros "
        "anteriores.",
    "Every temporal model keeps its baseline's backbone and head and only adds a fusion step between them. It is "
    "compared with its own baseline on the same seeds.":
        "Todo modelo temporal mantém o backbone e a cabeça da sua baseline e só acrescenta uma etapa de fusão entre "
        "eles. Ele é comparado com a própria baseline nas mesmas sementes.",
    "Three questions separate the temporal models: do they align the frames before fusing (v0.5, v0.6), how do "
    "they remember (a fixed 3-frame window for v0.2 to v0.6, a recurrent state for v0.7 and lite v0.6), and what "
    "do they cost (UFLD about 7 GMACs per frame, lite about 2).":
        "Três perguntas separam os modelos temporais: alinham os quadros antes de fundir (v0.5, v0.6)? Como "
        "lembram (janela fixa de 3 quadros da v0.2 à v0.6, estado recorrente na v0.7 e no lite v0.6)? Quanto "
        "custam (UFLD cerca de 7 GMACs por quadro, lite cerca de 2)?",
    "With feature caching each frame is encoded once, so a temporal model costs its baseline plus the fusion step, "
    "not three backbones.":
        "Com as features em cache, cada quadro é codificado uma vez; assim, um modelo temporal custa a sua baseline "
        "mais a etapa de fusão, e não três backbones.",
    "The controls are not candidates. +CT separates a temporal gain from extra training, lite v0.5 static "
    "separates it from the extra fusion layers, and the Kalman tracker shows what simple output filtering already "
    "gives.":
        "Os controles não são candidatos. O +CT separa o ganho temporal do treino extra, o lite v0.5 static o separa "
        "das camadas extras de fusão, e o filtro de Kalman mostra o que uma simples filtragem da saída já oferece.",
    "Shared skeleton of every model: frames -> backbone -> fusion -> head -> lanes.":
        "Esqueleto comum a todos os modelos: quadros -> backbone -> fusão -> cabeça -> faixas.",
    "t (current)": "t (atual)", "camera frames (step 2)": "quadros da câmera (passo 2)", "same weights": "mesmos pesos",
    "for every frame": "para todo quadro", "past frames: cached features": "quadros passados: features em cache",
    "Temporal fusion": "Fusão temporal", "where the models differ": "onde os modelos diferem",
    "baselines, +CT": "baselines, +CT", "weighted sum": "soma ponderada", "per-cell gates": "gates por célula",
    "warp + gate": "warp + gate", "Row-anchor": "Âncoras de linha", "head": "cabeça",
    "18 rows x 100": "18 linhas x 100", "cells per lane": "células por faixa", "lanes": "faixas",
    "Role": "Papel", "Frames used": "Quadros usados", "How history enters": "Como o histórico entra",
    "Aligns motion": "Alinha o movimento", "Memory": "Memória", "Params (M)": "Parâmetros (M)",
    "UFLD family (ResNet-18, ImageNet)": "Família UFLD (ResNet-18, ImageNet)",
    "Lite family (4-block CNN, from scratch)": "Família lite (CNN de 4 blocos, treinada do zero)",
    "UFLD family": "Família UFLD", "Lite family": "Família lite",
    "baseline": "baseline", "control": "controle", "temporal": "temporal",
    "weighted average": "média ponderada", "3-frame window": "janela de 3 quadros", "recurrent state": "estado recorrente",
    "yes (learned flow)": "sim (fluxo aprendido)", "no": "não", "n/a": "n/d",
    "warp + gate on copies of t": "warp + gate sobre cópias de t",
    "What it does": "O que faz", "Difference": "Diferença",
    "Official UFLD (Qin et al., ECCV 2020), supervisor's notebook":
        "UFLD oficial (Qin et al., ECCV 2020), notebook do orientador",
    "v0.4 control": "controle da v0.4", "Supervisor's notebook": "Notebook do orientador",
    "New in v0.4 (port of lite v0.5)": "Novo na v0.4 (adaptado do lite v0.5)", "New in v0.4": "Novo na v0.4",
    "ELAS development script": "Script de desenvolvimento do ELAS",
    "ELAS development script (refactored)": "Script de desenvolvimento do ELAS (refatorado)",
    "Ultra-Fast Lane Detection with an ImageNet-pretrained ResNet-18. Lane detection is row-wise classification: "
    "for each of 18 image rows and each lane slot the head picks one of 100 horizontal cells, or \"no lane\". One "
    "frame in, no memory.":
        "Ultra-Fast Lane Detection com uma ResNet-18 pré-treinada no ImageNet. A detecção de faixas é uma "
        "classificação por linha: para cada uma das 18 linhas da imagem e cada posição de faixa, a cabeça escolhe "
        "uma de 100 células horizontais, ou \"sem faixa\". Um quadro de entrada, sem memória.",
    "The reference of the UFLD family. Every UFLD temporal model starts from its trained weights and is compared "
    "with it on the same seeds.":
        "A referência da família UFLD. Todo modelo temporal UFLD parte dos pesos treinados dela e é comparado com "
        "ela nas mesmas sementes.",
    "The same network trained a second time from its own best checkpoint, with exactly the schedule the temporal "
    "variants get after their warm start.":
        "A mesma rede treinada uma segunda vez a partir do seu melhor checkpoint, exatamente com o cronograma que as "
        "variantes temporais recebem após a inicialização.",
    "A control, not a candidate. A temporal model that beats the baseline but not this control gained from extra "
    "training, not from time.":
        "Um controle, não um candidato. Um modelo temporal que supera a baseline mas não este controle ganhou pelo "
        "treino extra, não pelo tempo.",
    "Encodes the three frames with the shared ResNet and averages the feature maps with three learned weights "
    "(starting at 0.1 / 0.2 / 0.7) before the UFLD head. Trained with an extra existence loss and a "
    "temporal-consistency loss.":
        "Codifica os três quadros com a ResNet compartilhada e faz a média dos mapas de features com três pesos "
        "aprendidos (começando em 0.1 / 0.2 / 0.7) antes da cabeça da UFLD. Treinado com uma perda extra de "
        "existência e uma perda de consistência temporal.",
    "The simplest fusion: one weight per frame, the same at every pixel. Lane markings that moved between frames "
    "are averaged at different positions.":
        "A fusão mais simples: um peso por quadro, o mesmo em todos os pixels. Marcações que se moveram entre os "
        "quadros são somadas em posições diferentes.",
    "Like v0.2, but a small convolutional network predicts a softmax gate per frame and per feature cell, biased "
    "towards the current frame.":
        "Como a v0.2, mas uma pequena rede convolucional prevê um gate softmax por quadro e por célula de features, "
        "com viés para o quadro atual.",
    "Can use history only where the current frame is weak, such as an occluded cell. Like v0.2 it fuses feature "
    "maps that are not aligned.":
        "Pode usar o histórico só onde o quadro atual é fraco, como numa célula ocluída. Como a v0.2, funde mapas "
        "de features não alinhados.",
    "v0.2's network with a different loss: a soft-argmax coordinate term penalises the lane position error in "
    "pixels, and the existence loss is re-weighted.":
        "A rede da v0.2 com outra perda: um termo de coordenada via soft-argmax penaliza o erro de posição da faixa "
        "em pixels, e a perda de existência é reponderada.",
    "Same fusion as v0.2. It tests whether a position-aware loss helps; the network itself is unchanged.":
        "A mesma fusão da v0.2. Testa se uma perda que considera a posição ajuda; a rede em si não muda.",
    "A learned flow field (up to 4 cells of the 12x16 feature grid, about 128 px) warps each history map onto the "
    "current frame. The aligned history is compressed and blended in through a per-cell gate that starts almost "
    "closed.":
        "Um campo de fluxo aprendido (até 4 células da grade de features 12x16, cerca de 128 px) desloca cada mapa "
        "do histórico sobre o quadro atual. O histórico alinhado é comprimido e misturado por um gate por célula "
        "que começa quase fechado.",
    "The only UFLD model that aligns frames before fusing, so moving markings are not blurred. It adds about 1.7 M "
    "parameters to the baseline.":
        "O único modelo UFLD que alinha os quadros antes de fundir, para que marcações em movimento não fiquem "
        "borradas. Acrescenta cerca de 1.7 M de parâmetros à baseline.",
    "A convolutional GRU (64 channels) runs over the reduced feature maps from the oldest frame to the current "
    "one. Its final state is projected back and added to the current features. The projection starts at zero, so "
    "the untrained model is exactly the baseline.":
        "Uma GRU convolucional (64 canais) percorre os mapas de features reduzidos do quadro mais antigo ao atual. "
        "O estado final é projetado de volta e somado às features atuais. A projeção começa em zero, então o "
        "modelo não treinado é exatamente a baseline.",
    "Remembers through a gated recurrence instead of a weighted sum. A deployment can carry one hidden state per "
    "camera instead of storing past feature maps.":
        "Lembra por meio de uma recorrência com gates em vez de uma soma ponderada. Em produção, pode manter um "
        "estado por câmera em vez de guardar mapas de features anteriores.",
    "A small CNN: four stride-2 convolution blocks (32 to 128 channels) and a head that pools the features to the "
    "18x100 row-anchor grid and classifies every row with a shared MLP. Trained from scratch, without ImageNet.":
        "Uma CNN pequena: quatro blocos convolucionais com stride 2 (32 a 128 canais) e uma cabeça que reduz as "
        "features para a grade 18x100 de âncoras de linha e classifica cada linha com um MLP compartilhado. "
        "Treinada do zero, sem ImageNet.",
    "The reference of the lite family: about 8 times fewer parameters and 3 to 4 times less compute than UFLD.":
        "A referência da família lite: cerca de 8 vezes menos parâmetros e 3 a 4 vezes menos computação que a UFLD.",
    "The lite baseline trained a second time from its own best checkpoint with the temporal variants' schedule.":
        "A lite baseline treinada uma segunda vez a partir do seu melhor checkpoint, com o cronograma das variantes "
        "temporais.",
    "The equal-training control for lite v0.5 and lite v0.6.":
        "O controle com o mesmo treino para o lite v0.5 e o lite v0.6.",
    "The lite baseline plus warped residual fusion: a learned flow (up to 8 cells at stride 16, about 128 px) "
    "aligns each history feature map to the current one, and a per-pixel gate that starts closed mixes the "
    "aligned history in.":
        "A lite baseline com fusão residual alinhada: um fluxo aprendido (até 8 células com stride 16, cerca de "
        "128 px) alinha cada mapa de features do histórico ao atual, e um gate por pixel que começa fechado mistura "
        "o histórico alinhado.",
    "Same fusion idea as UFLD v0.6 on a network that is 8 times smaller.":
        "A mesma ideia de fusão da UFLD v0.6 numa rede 8 vezes menor.",
    "Lite v0.5 exactly (same layers, warm start, loss and budget), but every history frame is replaced by the "
    "current frame, in training and at test time.":
        "Exatamente o lite v0.5 (mesmas camadas, inicialização, perda e orçamento), mas todo quadro do histórico é "
        "trocado pelo quadro atual, no treino e no teste.",
    "A capacity control. Lite v0.5 has extra fusion layers; if it beats this model, the gain comes from the "
    "earlier frames and not from the extra layers.":
        "Um controle de capacidade. O lite v0.5 tem camadas extras de fusão; se ele superar este modelo, o ganho vem "
        "dos quadros anteriores e não das camadas extras.",
    "The lite baseline plus the ConvGRU fusion of UFLD v0.7 (64 hidden channels, read-out starting at zero).":
        "A lite baseline com a fusão ConvGRU da UFLD v0.7 (64 canais ocultos, saída começando em zero).",
    "Recurrent memory on the smallest network: the candidate for an embedded system that keeps one state tensor "
    "per camera. Unlike v0.5 it does not align frames explicitly.":
        "Memória recorrente na menor rede: a candidata para um sistema embarcado que mantém um tensor de estado por "
        "câmera. Ao contrário da v0.5, não alinha os quadros explicitamente.",
    "Reference, not a model:": "Referência, não um modelo:",
    "Any model + Kalman tracker: a causal constant-velocity Kalman filter on every lane point, with existence "
    "smoothing, tuned on validation. It costs microseconds per frame and is the bar a learned temporal model has "
    "to clear.":
        "Qualquer modelo + filtro de Kalman: um filtro de Kalman causal de velocidade constante em cada ponto de "
        "faixa, com suavização da existência, ajustado na validação. Custa microssegundos por quadro e é o patamar "
        "que um modelo temporal aprendido precisa superar.",
    # results
    "Lane F1 on unseen roads": "Lane F1 em estradas nunca vistas",
    "CULane-style lane F1: each lane drawn 12 px wide (30 px at 1640 px, scaled to 640 px), matched one-to-one, "
    "correct when IoU ≥ 0.5. Post-processing tuned on validation only. On ELAS every model predicts both ego lanes "
    "in every frame, so precision = recall = F1: the errors are misplaced lanes, not missed ones.":
        "Lane F1 no estilo CULane: cada faixa desenhada com 12 px de largura (30 px em 1640 px, escalado para "
        "640 px), pareamento um a um, correta quando IoU ≥ 0.5. Pós-processamento ajustado só na validação. No ELAS "
        "todo modelo prevê as duas faixas do veículo em todo quadro, então precisão = revocação = F1: os erros são "
        "faixas mal posicionadas, não faixas perdidas.",
    "Held-out test scenes (unseen roads)": "Cenas de teste (estradas nunca vistas)",
    "Seen-scene test blocks": "Blocos de teste de cenas já vistas",
    "bars = mean over seeds, dots = individual seeds": "barras = média das sementes, pontos = cada semente",
    "Held-out test scenes, all metrics (mean ± std over seeds)":
        "Cenas de teste, todas as métricas (média ± desvio padrão entre sementes)",
    "Held-out test scenes (mean ± std over seeds)": "Cenas de teste (média ± desvio padrão entre sementes)",
    "Model": "Modelo", "Lane F1 @0.5": "Lane F1 @0.5", "Lane F1 @0.35": "Lane F1 @0.35", "Pixel F1": "F1 de pixel",
    "Anchor F1": "F1 de âncora", "Jitter px": "Jitter (px)", "Seen-scene test": "Teste em cenas já vistas",
    "Does temporal context help?": "O contexto temporal ajuda?",
    "Temporal model minus its single-frame baseline (held-out lane F1)":
        "Modelo temporal menos a sua baseline de um quadro (lane F1 no teste)",
    "Temporal model": "Modelo temporal", "Compared with": "Comparado com", "Seeds": "Sementes",
    "Δ lane F1": "Δ lane F1", "95 % CI": "IC 95 %", "Wilcoxon p": "p de Wilcoxon", "Holm p": "p de Holm", "Verdict": "Veredito",
    "underpowered": "poucas sementes", "significant": "significativo", "not significant": "não significativo",
    "Equal-training control": "Controle com o mesmo treino",
    "Temporal models start from their baseline's best checkpoint and train further. The +CT baseline gets the "
    "same extra training; a temporal gain that survives this comparison is not an effect of more epochs.":
        "Os modelos temporais partem do melhor checkpoint da sua baseline e treinam mais. A baseline +CT recebe o "
        "mesmo treino extra; um ganho temporal que resiste a essa comparação não é efeito de mais épocas.",
    "Capacity control: extra layers or earlier frames?": "Controle de capacidade: camadas extras ou quadros anteriores?",
    "Lite v0.5 static is lite v0.5 with every history frame replaced by the current frame, in training and at "
    "test time: the same layers, warm start and budget, but no temporal information. \"Lite v0.5 − static\" is "
    "what the earlier frames add; \"static − baseline\" is what the extra fusion layers add on their own.":
        "O lite v0.5 static é o lite v0.5 com todo quadro do histórico trocado pelo quadro atual, no treino e no "
        "teste: mesmas camadas, inicialização e orçamento, mas sem informação temporal. \"Lite v0.5 − static\" é o "
        "que os quadros anteriores acrescentam; \"static − baseline\" é o que as camadas extras de fusão acrescentam "
        "sozinhas.",
    "Against the Kalman tracker": "Contra o filtro de Kalman",
    "The same predictions filtered by the output Kalman tracker (tuned on validation). This is what time gives "
    "almost for free; a learned temporal model should beat its baseline + Kalman (paired table below).":
        "As mesmas previsões filtradas pelo filtro de Kalman na saída (ajustado na validação). É o que o tempo dá "
        "quase de graça; um modelo temporal aprendido deveria superar a sua baseline + Kalman (tabela pareada "
        "abaixo).",
    "Lane F1": "Lane F1", "Lane F1 + Kalman": "Lane F1 + Kalman", "Jitter px + Kalman": "Jitter (px) + Kalman",
    "When the current frame is degraded": "Quando o quadro atual está degradado",
    "The held-out test scenes again, but every current frame is corrupted (an occluding box, strong blur, "
    "darkening or noise; the same corruption for every model) while the earlier frames stay clean. This is where "
    "earlier frames should pay off: a single-frame model only sees the damaged image. Validation-tuned "
    "post-processing, nothing re-tuned.":
        "As mesmas cenas de teste, mas todo quadro atual é corrompido (um retângulo cobrindo a imagem, desfoque "
        "forte, escurecimento ou ruído; a mesma corrupção para todos os modelos) enquanto os anteriores ficam "
        "limpos. É aqui que os quadros anteriores deveriam compensar: um modelo de um quadro só vê a imagem "
        "danificada. Pós-processamento ajustado na validação, nada reajustado.",
    "Held-out lane F1 with a degraded current frame": "Lane F1 no teste com o quadro atual degradado",
    "Clean": "Limpo", "Degraded": "Degradado", "occlude": "oclusão", "blur": "desfoque", "darken": "escurecimento",
    "noise": "ruído", "Degraded + Kalman": "Degradado + Kalman",
    "Recurrent models with a carried state": "Modelos recorrentes com estado contínuo",
    "Held-out lane F1 when the ConvGRU keeps one state per stream across the whole scene, minus the window mode "
    "used in training (a zero state at every clip).":
        "Lane F1 no teste quando a ConvGRU mantém um estado por sequência ao longo da cena inteira, menos o modo de "
        "janela usado no treino (estado zero a cada clipe).",
    "Carry − window": "Contínuo − janela",
    "Is it the history?": "É o histórico?",
    "Test-time ablation: history frames replaced by the current frame. The gain is what the real history "
    "contributes to each trained model.":
        "Ablação no teste: quadros do histórico trocados pelo quadro atual. O ganho é o que o histórico real "
        "contribui para cada modelo treinado.",
    "F1 gain from real history": "Ganho de F1 com o histórico real",
    "Where temporal information should help": "Onde a informação temporal deveria ajudar",
    "Held-out lane F1 per scene condition from the ELAS scene tags (whole scenes, so these are three scenes, not "
    "frame-level subsets) and jitter, the mean frame-to-frame change of the predicted lane position (lower is "
    "steadier; it also contains real lane motion).":
        "Lane F1 no teste por condição de cena, segundo as marcações de cena do ELAS (cenas inteiras, portanto três "
        "cenas, não subconjuntos de quadros), e jitter, a variação média da posição prevista da faixa entre quadros "
        "(menor é mais estável; inclui também o movimento real da faixa).",
    "occlusion": "oclusão", "transition": "transição", "jitter px": "jitter (px)",
    "The lite family with a longer budget": "A família lite com orçamento maior",
    "Lite family, v0.4 recipe, up to 16 epochs (patience 4), 2 seeds, no HPO. Indicative only.":
        "Família lite, receita da v0.4, até 16 épocas (paciência 4), 2 sementes, sem busca de hiperparâmetros. "
        "Apenas indicativo.",
    "Head to head on the held-out roads": "Frente a frente nas estradas de teste",
    ", the others from": ", os demais de",
    # augmentation / history ablations
    "Fixing the UFLD overfitting": "Corrigindo o overfitting da UFLD",
    "The first pilot showed every UFLD run peaking at epoch 1. Each arm below changes one thing in the training of "
    "the UFLD baseline (same split, seeds, epochs and evaluation) and is compared with the original recipe "
    "(photometric augmentation). The recipe for later runs is chosen on the validation column; the held-out column "
    "is reported, never used for choosing. \"Best epochs\" are the validation-selected epochs per seed; the \"more "
    "scenes\" arm has its own, larger validation set. Three seeds cannot reach significance, so the verdicts are "
    "indicative.":
        "O primeiro piloto mostrou toda execução da UFLD com o melhor resultado na época 1. Cada braço abaixo muda "
        "uma coisa no treino da UFLD baseline (mesma divisão, sementes, épocas e avaliação) e é comparado com a "
        "receita original (aumento fotométrico). A receita das execuções seguintes é escolhida pela coluna de "
        "validação; a coluna de teste é apenas relatada, nunca usada para escolher. \"Melhores épocas\" são as "
        "épocas escolhidas na validação em cada semente; o braço \"mais cenas\" tem o seu próprio conjunto de "
        "validação, maior. Três sementes não alcançam significância, então os vereditos são indicativos.",
    "Held-out lane F1 per arm": "Lane F1 no teste por braço",
    "Arm": "Braço", "Validation lane F1": "Lane F1 na validação", "Held-out lane F1": "Lane F1 no teste",
    "Best epochs": "Melhores épocas", "Held-out Δ vs reference": "Δ no teste × referência",
    "How far back to look": "Até onde olhar para trás",
    "History-length ablation: every arm uses the same split (purge gap large enough for the longest history) and "
    "the same trained baseline, so only the temporal context changes. Cells are means over seeds on the held-out "
    "scenes; \"not run\" arms can be added with":
        "Ablação do comprimento do histórico: todo braço usa a mesma divisão (intervalo de descarte grande o bastante "
        "para o histórico mais longo) e a mesma baseline treinada, então só o contexto temporal muda. As células são "
        "médias das sementes nas cenas de teste; braços \"não rodados\" podem ser adicionados com",
    "frames in the clip": "quadros no clipe", "frame step (history frame k is t - k x step)":
        "passo entre quadros (o quadro k do histórico é t - k x passo)",
    "frame step": "passo entre quadros", "not run": "não rodado",
    "Training curves": "Curvas de treino",
    "Temporal models and +CT controls start from their family baseline's best checkpoint.":
        "Os modelos temporais e os controles +CT partem do melhor checkpoint da baseline da sua família.",
    "validation lane F1 per epoch; solid = first seed, dashed = second, dot = best epoch (the checkpoint kept)":
        "lane F1 de validação por época; contínua = primeira semente, tracejada = segunda, ponto = melhor época (o "
        "checkpoint mantido)",
    # demo, latency, export
    "Streaming on a held-out road": "Inferência contínua numa estrada de teste",
    "240 consecutive frames of scene BR_S02 (never seen in training), run frame by frame through the streaming "
    "detector: left UFLD baseline, right Lite v0.5 with cached history features (first pilot checkpoints). Red = "
    "ego-left slot, yellow = ego-right slot, numbers = mean existence probability of each lane.":
        "240 quadros consecutivos da cena BR_S02 (nunca vista no treino), processados quadro a quadro pelo detector "
        "contínuo: à esquerda a UFLD baseline, à direita o lite v0.5 com features do histórico em cache "
        "(checkpoints do primeiro piloto). Vermelho = faixa esquerda do veículo, amarelo = faixa direita, números = "
        "probabilidade média de existência de cada faixa.",
    "Latency and real-time budgets": "Latência e orçamentos de tempo real",
    "preprocess": "pré-processamento", "model": "modelo", "postprocess": "pós-processamento",
    "Simulated 30 FPS camera: service times = measured × slowdown. The Jetson slowdowns are assumptions from "
    "peak-throughput ratios, not measurements on a Jetson. \"History fallbacks\" are frames whose history frames "
    "were skipped because the worker was busy.":
        "Câmera simulada a 30 FPS: tempos de serviço = medido × fator de lentidão. Os fatores da Jetson são "
        "suposições a partir de razões de desempenho de pico, não medições numa Jetson. \"Histórico substituído\" "
        "são quadros cujo histórico foi pulado porque o processamento estava ocupado.",
    "Profile": "Perfil", "Slowdown": "Lentidão", "Mode": "Modo", "Processed FPS": "FPS processados",
    "p95 latency ms": "Latência p95 (ms)", "History fallbacks": "Histórico substituído", "Budgets": "Orçamentos",
    "single": "quadro único", "cached": "em cache", "recompute": "recalculado", "met": "atendido",
    "missed": "não atendido",
    "Export and precision checks": "Exportação e verificação de precisão",
    "Each exported model runs the same frames as PyTorch through the same streaming code (first pilot "
    "checkpoints). Val lane F1 is measured on 240 validation frames with the exact training-time temporal context; "
    "the difference to PyTorch FP32 is in brackets. TensorRT ran on the development GPU, not on a Jetson.":
        "Cada modelo exportado processa os mesmos quadros que o PyTorch pelo mesmo código de inferência contínua "
        "(checkpoints do primeiro piloto). O lane F1 de validação é medido em 240 quadros de validação com o mesmo "
        "contexto temporal do treino; a diferença para o PyTorch FP32 está entre parênteses. O TensorRT rodou na GPU "
        "de desenvolvimento, não numa Jetson.",
    "Backend": "Backend", "Val lane F1": "Lane F1 (val.)", "Numerics": "Numérica",
    "max |Δ logits|": "máx. |Δ logits|", "max |Δ exist|": "máx. |Δ existência|",
    "mean lane Δx px": "Δx médio da faixa (px)", "4-block CNN (lite)": "CNN de 4 blocos (lite)",
    "within tolerance": "dentro da tolerância", "outside tolerance": "fora da tolerância", "not built": "não gerado",
    "Examples from the held-out scenes": "Exemplos das cenas de teste",
    # roadmap, reproduce, footer
    "Roadmap": "Roteiro",
    "First make a lightweight temporal model beat the UFLD baseline under the full protocol; only then optimise "
    "for an in-car embedded system.":
        "Primeiro, fazer um modelo temporal leve superar a UFLD baseline no protocolo completo; só depois otimizar "
        "para um sistema embarcado no carro.",
    "done": "feito", "in progress": "em andamento", "next": "próximo", "later": "depois",
    "Reproduce and continue": "Reproduzir e continuar",
    "The full experiment differs from the pilots in seeds (6 instead of 2), epochs (up to 50 with patience 10) and "
    "hyper-parameter search (the same budget for every model). Only the full run can support claims.":
        "O experimento completo difere dos pilotos nas sementes (6 em vez de 2), nas épocas (até 50, com paciência "
        "10) e na busca de hiperparâmetros (o mesmo orçamento para todos os modelos). Só a execução completa pode "
        "sustentar conclusões.",
    ". Numbers come from the run folders; the findings, the roadmap and the model descriptions are written by hand.":
        ". Os números vêm das pastas das execuções; os resultados comentados, o roteiro e as descrições dos modelos "
        "são escritos à mão.",
}

PATTERNS: list[tuple[str, str]] = [
    (r"^(\d+) models$", r"\1 modelos"),
    (r"^(\d+) seeds$", r"\1 sementes"),
    (r"^held-out scenes (.+)$", r"cenas de teste \1"),
    (r"^GPU time (.+)$", r"tempo de GPU \1"),
    (r"^UFLD baseline vs the best temporal model \((.+)\): within seed noise \((\d+) seeds\)$",
     r"UFLD baseline × melhor modelo temporal (\1): dentro do ruído entre sementes (\2 sementes)"),
    (r"^UFLD baseline vs (.+) when the current frame is occluded and the earlier frames are clean: (\S+), in every seed$",
     r"UFLD baseline × \1 com o quadro atual ocluído e os anteriores limpos: \2, em todas as sementes"),
    (r"^UFLD baseline vs (.+) when the current frame is occluded and the earlier frames are clean: (\S+), not in every seed$",
     r"UFLD baseline × \1 com o quadro atual ocluído e os anteriores limpos: \2, não em todas as sementes"),
    (r"^−(\d+) % jitter$", r"−\1 % de jitter"),
    (r"^Lite v0\.5 lane F1 and parameters, against (\S+) with (\S+) M for the UFLD baseline \((\d+)× fewer parameters\)$",
     r"Lane F1 e parâmetros do lite v0.5, contra \1 com \2 M da UFLD baseline (\3× menos parâmetros)"),
    (r"^Held-out lane F1 \(clean frames\) and parameters\. Compute per frame: UFLD (\S+) GMAC, lite (\S+) GMAC; a temporal "
     r"model streaming with cached features adds only its fusion \(it recomputes 3 frames otherwise\)\. Lite models from$",
     r"Lane F1 no teste (quadros limpos) e parâmetros. Computação por quadro: UFLD \1 GMAC, lite \2 GMAC; um modelo "
     r"temporal com features em cache acrescenta só a sua fusão (caso contrário, recalcula 3 quadros). Modelos lite da execução"),
    (r"^4 · How far back to look \((.+)\)$", r"4 · Até onde olhar para trás (\1)"),
    (r"^Held-out lane F1 of (.+) minus the UFLD baseline for each history length and spacing \(history frame k = "
     r"t − k·step; same split and baseline for every arm\)\. Spacing changes little; five frames are worse than three\.$",
     r"Lane F1 no teste da \1 menos a UFLD baseline para cada comprimento e espaçamento do histórico (quadro k do "
     r"histórico = t − k·passo; mesma divisão e baseline em todos os braços). O espaçamento muda pouco; cinco "
     r"quadros são piores que três."),
    (r"^(\d+) frames, step (\d+)$", r"\1 quadros, passo \2"),
    (r"^(.+) · single$", r"\1 · quadro único"),
    (r"^(.+) · cached$", r"\1 · em cache"),
    (r"^(.+) · recompute$", r"\1 · recalculado"),
    (r"^(.+) \((\d+) ep\)$", r"\1 (\2 ép.)"),
    (r"^(\S+) M params · (\S+) GMACs per clip, (\d+) frames per clip$",
     r"\1 M parâmetros · \2 GMACs por clipe, \3 quadros por clipe"),
    (r"^(\S+) M params · (\S+) GMACs per clip$", r"\1 M parâmetros · \2 GMACs por clipe"),
    (r"^Each temporal model is compared with the single-frame model of its own family on the same seeds \(exact "
     r"Wilcoxon, Holm correction\)\. With (\d+) seeds? no difference can reach p < 0\.05 unless there are at least 6 "
     r"seeds, so the verdicts here are indicative\.$",
     r"Cada modelo temporal é comparado com o modelo de um quadro da sua família nas mesmas sementes (Wilcoxon "
     r"exato, correção de Holm). Com \1 semente(s), nenhuma diferença pode alcançar p < 0.05; são precisas pelo "
     r"menos 6 sementes, então os vereditos aqui são indicativos."),
    (r"^Same split, recipe and evaluation; the numbers in brackets are the maximum epochs of each run \(early "
     r"stopping may end sooner\)\. The (.+) models come from$",
     r"Mesma divisão, receita e avaliação; os números entre parênteses são as épocas máximas de cada execução (a "
     r"parada antecipada pode terminar antes). Os modelos \1 vêm de"),
    (r"^(.+): lane F1 gain over (.+)$", r"\1: ganho de lane F1 sobre \2"),
    (r"^(.+): held-out lane F1$", r"\1: lane F1 no teste"),
    (r"^Measured at batch 1 on (.+) with 640×480 frames of the held-out clip \(first pilot checkpoints\)\. "
     r"\"cached\" reuses the stored features of previous frames; \"recompute\" re-encodes all three frames every "
     r"time\. Preprocessing \(a PIL bilinear resize on the CPU, identical to training\) is the largest cost\.$",
     r"Medido com lote 1 na \1, com quadros 640×480 do clipe de teste (checkpoints do primeiro piloto). \"em "
     r"cache\" reaproveita as features guardadas dos quadros anteriores; \"recalculado\" codifica os três quadros "
     r"toda vez. O pré-processamento (redimensionamento bilinear do PIL na CPU, igual ao do treino) é o maior custo."),
    (r"^(.+), held-out test, a typical success: original \| ground truth \(green\) \| prediction \(red\) - (.+)$",
     r"\1, teste, um acerto típico: original | anotação (verde) | previsão (vermelho) - \2"),
    (r"^(.+), held-out test, a typical error: original \| ground truth \(green\) \| prediction \(red\) - (.+)$",
     r"\1, teste, um erro típico: original | anotação (verde) | previsão (vermelho) - \2"),
    (r"^Generated (\S+) from git (\S+) by$", r"Gerado em \1 a partir do git \2 por"),
    (r"^Authors: (.+)$", r"Autores: \1"),
    (r"^(.+ · .+), Brazil$", r"\1, Brasil"),
]
_COMPILED = [(re.compile(p), r) for p, r in PATTERNS]
_SKIP = {"pre", "code", "style", "script", "title"}
_PORTUGUESE = re.compile(r"[ãõçáéíóúêâ]|\b(que|não|os|das|dos|uma|com)\b")
_ENGLISH = re.compile(r"\b(the|and|with|from|of|is|are|to|on|for|by|every|when|than|without|only)\b")


def _decimal_comma(text: str) -> str:
    """0.881 -> 0,881, except inside names such as v0.3 or file versions."""
    return re.sub(r"(?<![\w.])(\d+)\.(\d+)", r"\1,\2", text)


def _translate_text(core: str) -> str | None:
    if core in EXACT:
        return EXACT[core]
    for rx, repl in _COMPILED:
        if rx.match(core):
            return rx.sub(repl, core)
    return None


def translate_html(page: str) -> tuple[str, list[str]]:
    """Translate the visible text of a generated page; return it and the text
    nodes that still look English (for the coverage test)."""
    # English halves of bilingual blocks are dropped on the Portuguese page
    page = re.sub(r'<span class="en(?: [^"]*)?"[^>]*>.*?</span>', "", page, flags=re.S)
    parts = re.split(r"(<[^>]+>)", page)
    stack: list[str] = []
    missing: list[str] = []
    out = []
    for part in parts:
        if part.startswith("<"):
            m = re.match(r"<(/?)([a-zA-Z0-9]+)", part)
            if m:
                name = m.group(2).lower()
                if m.group(1):
                    if name in _SKIP and name in stack:
                        stack.remove(name)
                elif name in _SKIP and not part.endswith("/>"):
                    stack.append(name)
            out.append(part)
            continue
        if not part.strip() or stack:
            out.append(part)
            continue
        lead = part[: len(part) - len(part.lstrip())]
        trail = part[len(part.rstrip()):]
        core = html.unescape(part.strip())
        translated = _translate_text(core)
        if translated is None:
            translated = core
            if _ENGLISH.search(core) and re.search(r"[a-z]{3,}", core) and not _PORTUGUESE.search(core):
                missing.append(core)
        out.append(lead + html.escape(_decimal_comma(translated), quote=False) + trail)
    return "".join(out), missing
