# Research summary: challenges and next steps

Shown in the "Where the research stands" section of the results page, under the key figures and charts. Those numbers are computed from the run folders; the text here is written by hand (30 Sep 2026). Only lines starting with `- ` or `1. ` under a `## ` heading are shown.

## Desafios

- **Base de dados pequena.** O protocolo usa 10 cenas do ELAS, e só 3 ficam reservadas para teste. A UFLD decorava a posição das faixas até entrar o aumento de dados geométrico.
- **Quadros limpos escondem o efeito temporal.** Com quadros limpos, os modelos temporais UFLD ficam a cerca de ±0,02 de lane F1 da baseline. Duas sementes não bastam para separar isso do ruído.
- **Os modelos leves treinam devagar e de forma instável.** Treinados do zero, precisam de 16 épocas ou mais. O lite v0.5 sem histórico real (o controle de capacidade) marcou 0,877 em uma semente e 0,627 na outra. Com quadros limpos, ainda não se sabe quanto do ganho vem dos quadros anteriores; com o quadro atual ocluído, o histórico vale +0,17 e +0,38 nas duas sementes.
- **Memória em sequências longas.** Os modelos recorrentes derivam quando o estado é mantido ao longo de uma cena inteira (lite v0.6: −0,48 de lane F1). Eles foram treinados só com clipes de 3 quadros.
- **Mais histórico não é melhor.** Espaçar os quadros do histórico de 2, 5 ou 10 quadros não muda nada; usar 5 quadros em vez de 3 reduz o lane F1.
- **Uma única base de dados até agora.** CULane, TuSimple e OpenLane ainda não foram baixadas. Os leitores estão prontos e foram testados em cópias sintéticas da estrutura delas.
- **Ainda sem medição embarcada.** Os números da Jetson são simulações com fatores de lentidão assumidos, não medições.

## Próximos passos

1. Rodar o protocolo completo no ELAS: 6 sementes, até 50 épocas e o mesmo orçamento de busca de hiperparâmetros para todos os modelos (cerca de 4–5 dias na RTX 3050).
2. Tornar a robustez um resultado principal: primeiro as corrupções sintéticas, depois oclusões naturais, noite e chuva.
3. Baixar CULane e TuSimple (o guia de download está pronto), pré-treinar nelas, ajustar no ELAS e reportar as métricas oficiais delas.
4. Resolver a questão dos modelos leves: mais sementes e treino mais longo para o lite v0.5 contra seu controle de capacidade, e treinar os modelos recorrentes com sequências longas.
5. Última fase, embarcado: exportar o melhor modelo temporal leve para TensorRT (FP16/INT8) e medir latência, memória e consumo de energia em uma Jetson.

## Challenges

- **Small dataset.** The protocol uses 10 ELAS scenes, and only 3 are held out for testing. UFLD memorised lane positions until geometric augmentation was added.
- **Clean frames hide the temporal effect.** On clean held-out frames the temporal UFLD models stay within about ±0.02 lane F1 of the baseline. Two seeds cannot separate that from noise.
- **The lightweight models train slowly and unstably.** Trained from scratch, they need 16 epochs or more. Lite v0.5 without real history (the capacity control) scored 0.877 in one seed and 0.627 in the other. On clean frames it is still open how much of the gain comes from the earlier frames; with an occluded current frame, the history is worth +0.17 and +0.38 in both seeds.
- **Memory over long sequences.** The recurrent models drift when their state is carried across a whole scene (lite v0.6: −0.48 lane F1). They were trained on 3-frame clips only.
- **More history is not better.** Spacing the history frames 2, 5 or 10 frames apart changes nothing; 5 frames instead of 3 costs lane F1.
- **One dataset so far.** CULane, TuSimple and OpenLane are not downloaded yet. Their readers are ready and were tested on synthetic copies of their layouts.
- **No embedded measurement yet.** The Jetson numbers are simulations based on assumed slowdowns, not measurements.

## Next steps

1. Run the full protocol on ELAS: 6 seeds, up to 50 epochs, and the same hyper-parameter search budget for every model (about 4–5 days on the RTX 3050).
2. Make robustness a primary result: the synthetic corruptions now, then natural occlusions, night and rain frames.
3. Download CULane and TuSimple (the download guide is ready), pre-train on them, fine-tune on ELAS, and report their official metrics.
4. Settle the lightweight question: more seeds and longer training for lite v0.5 against its capacity control, and train the recurrent models on long sequences.
5. Last phase, embedded: export the best lightweight temporal model to TensorRT (FP16/INT8) and measure latency, memory and power on a Jetson.
