# Roteiro

Versão em português de `ROADMAP.md`, usada pela página em português (`python -m tac_ufld site --lang pt`). Marcadores lidos pela página: `[done]`, `[progress]`, `[next]`, `[later]`.

## Fase 1: protocolo e primeiro piloto (v0.3)
- [done] Protocolo do ELAS sem vazamento de dados: três cenas de estrada reservadas para teste, blocos de 60 quadros com intervalo de descarte, seis sementes, busca de hiperparâmetros com o mesmo orçamento para todos os modelos e seleção apenas na validação.
- [done] Piloto na GPU (2 sementes, até 4 épocas): o lite v0.5 foi o melhor nas estradas nunca vistas; toda execução da UFLD atinge o melhor resultado na época 1 e depois sofre overfitting; o histórico acrescenta pouco a qualquer modelo.
- [done] Ferramentas de implantação: ONNX, TensorRT FP32 / FP16 / INT8 na GPU do desktop e inferência contínua (streaming) com features do histórico em cache.

## Fase 2: fazer um modelo temporal leve superar a UFLD baseline (v0.4)
- [done] Corrigir o overfitting da UFLD: o aumento de dados geométrico leva a UFLD baseline de 0,07 para 0,86 de lane F1 no teste (`configs/ablations/augmentation.yaml`, escolhido na validação); aumento fotométrico estendido, espelhamento e dropout + label smoothing não ajudam.
- [done] As outras alavancas contra o overfitting, cada uma como mudança única na receita original: taxa de aprendizado do backbone 10× menor, camadas iniciais congeladas, cabeça menor, degradação do quadro atual e 8 cenas a mais do ELAS. Nenhuma resolve o overfitting sozinha (lane F1 no teste de 0,05 a 0,21, melhor época ainda 1 ou 2); o aumento geométrico continua sendo a receita. A inicialização a partir de outra base (`model.init_checkpoint`) está implementada para quando houver pesos de CULane / TuSimple.
- [done] Histórico mais longo (`configs/ablations/history.yaml`, UFLD v0.4, 4 braços): espaçamentos de 2, 5 e 10 quadros dão o mesmo ganho sobre a baseline (cerca de +0,03); 5 quadros em vez de 3 eliminam o ganho. Outros braços da grade (passos de 1 a 15, 2 a 5 quadros) podem ser adicionados com `ablate --only`.
- [done] Degradação do quadro atual: só o quadro atual é ocluído, borrado, escurecido ou recebe ruído, enquanto o histórico fica limpo, para que o modelo temporal precise usar o histórico (`data.augmentation.current_frame_prob`).
- [done] Alinhar antes de fundir: UFLD v0.6 (o warp e o gate aprendidos do lite v0.5 sobre as features da UFLD).
- [done] Estado recorrente: UFLD v0.7 e lite v0.6 (ConvGRU); a inferência contínua pode manter um estado por câmera (`--mode carry`).
- [done] Controles com o mesmo treino: as baselines +CT recebem o mesmo treino extra que os modelos temporais.
- [done] Avaliar onde o tempo deve ajudar: lane F1 por condição de cena, jitter e um filtro de Kalman ajustado na validação como referência temporal barata.
- [done] Segundo piloto com os modelos e a receita da v0.4 (`configs/elas_pilot_v2.yaml`): UFLD baseline 0,881, melhor UFLD temporal (v0.7) 0,895; os modelos lite precisam de mais épocas.
- [done] Controle de capacidade `lite_v05_static`: lite v0.5 alimentado com o quadro atual em todas as posições, para separar o ganho temporal das camadas extras de fusão.
- [done] Família lite com orçamento maior (até 16 épocas) e o controle de capacidade (`configs/elas_lite_long.yaml`): o lite v0.5 chega a 0,850 com 8× menos parâmetros que a UFLD; com o quadro atual ocluído, o histórico vale +0,17 e +0,38 sobre o controle sem histórico.
- [done] Avaliação de robustez: quadro atual ocluído, borrado, escurecido ou com ruído e histórico limpo. Com oclusão, a UFLD v0.3 mantém 0,875 de lane F1 contra 0,761 da baseline, nas duas sementes.
- [next] Treinar os modelos recorrentes com sequências longas, para que um único estado contínuo por câmera funcione (o modo contínuo deriva após treino com 3 quadros).

## Fase 3: protocolo completo (orientador)
- [next] Execução completa: 6 sementes, busca de hiperparâmetros, até 50 épocas, com a receita e os modelos escolhidos nos pilotos (`configs/elas.yaml`).
- [later] Pré-treino em CULane / TuSimple e ajuste fino no ELAS (precisa das bases ou dos pesos oficiais da UFLD).

## Fase 4: implantação embarcada automotiva (última etapa)
- [later] Medir no dispositivo real: gerar os engines TensorRT em uma Jetson e rodar o script de medição; os números atuais da Jetson são simulações com fatores de lentidão assumidos.
- [later] Tirar o pré-processamento da CPU: o redimensionamento na CPU leva cerca de 9 ms de um quadro de 14 a 19 ms. Redimensionar no processador de imagem da câmera ou na GPU com buffers sem cópia, e retreinar com esse mesmo redimensionamento.
- [later] Recortar a estrada: as âncoras de linha cobrem só os 41 % inferiores da imagem, então o recorte reduz o custo em cerca de 2,4 vezes; uma resolução de entrada menor é a segunda opção. As duas exigem retreino.
- [later] TensorRT FP16 por padrão (sem perda no piloto); INT8 só se o FP16 for lento demais, calibrado com a câmera do carro ou com treino ciente da quantização para os modelos lite.
- [later] Preferir a família lite: cerca de 8 vezes menos parâmetros e 3,6 vezes menos computação que a UFLD.
- [later] Robustez a quadros perdidos: treinar com quadros descartados e espaçar o histórico por tempo, pois a câmera do carro pode ter outra taxa de quadros que a do ELAS.
- [later] Execução de produção: TensorRT em C++ ou DeepStream em vez de Python, captura / inferência / pós-processamento em paralelo, sempre processar o quadro mais recente, clocks fixos e testes térmicos dentro do invólucro.
