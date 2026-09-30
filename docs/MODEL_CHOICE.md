# Which model to compare with the baseline

The hand-written part of the results-page section "Qual modelo comparar com a baseline / Which model to compare with the baseline" (30 Sep 2026). The tables of that section are computed from the run folders. This text holds no numbers, so it stays true as long as the tables keep telling the same story; re-read it after the full run. Portuguese first, as on the page. Model notes: `- \`variant\`: Portuguese || English`.

## Recomendação

- **Comparação principal: UFLD v0.3 (gated) × UFLD baseline.** Mesma rede e mesmo treino, com praticamente nenhum parâmetro a mais; qualquer diferença vem do uso dos quadros anteriores. É o modelo que mais mantém as faixas quando o quadro atual está ocluído, e empata com a baseline em quadros limpos.
- **Comparação secundária, para o embarcado: lite v0.5 × UFLD baseline, e lite v0.5 × o mesmo modelo sem histórico.** Mostra o custo de um modelo muito menor e, pelo controle sem histórico, que a robustez vem do tempo.
- **Declarar antes da execução completa.** Esta escolha foi feita olhando o teste; para a afirmação ser válida, declare agora as comparações principais e a métrica com oclusão como métrica principal, e use as 6 sementes da execução completa para confirmar ou rejeitar.
- **Condição para comparar as famílias:** treinar UFLD e lite com o mesmo protocolo (mesmas épocas máximas e busca de hiperparâmetros) antes de colocar os dois lado a lado num artigo.

## Recommendation

- **Primary comparison: UFLD v0.3 (gated) vs the UFLD baseline.** Same network and same training, with almost no extra parameters, so any difference comes from using the earlier frames. It is the model that keeps the lanes best when the current frame is occluded, and it ties the baseline on clean frames.
- **Secondary comparison, for the embedded goal: lite v0.5 vs the UFLD baseline, and lite v0.5 vs the same model without history.** It shows the cost of a much smaller model and, through the no-history control, that the robustness comes from time.
- **Declare before the full run.** This choice was made by looking at the test set. For the claim to hold, declare the primary comparisons now, with occluded-frame F1 as a primary metric, and let the 6 seeds of the full run confirm or reject them.
- **Condition for comparing the families:** train UFLD and lite under the same protocol (same maximum epochs and hyper-parameter search) before putting them side by side in a paper.

## Notas por modelo

- `ufld_baseline`: referência de um quadro || single-frame reference
- `ufld_baseline_ct`: controle: o mesmo treino extra dos modelos temporais, sem histórico || control: the temporal models' extra training, without history
- `ufld_v02`: sem ganho com oclusão || no gain under occlusion
- `ufld_v03`: **recomendado**: robusto à oclusão, quase sem custo extra || **recommended**: robust to occlusion, almost no extra cost
- `ufld_v04`: sem ganho com oclusão; escolhido na validação limpa para a ablação de histórico || no gain under occlusion; chosen on clean validation for the history ablation
- `ufld_v06`: quase tão robusto quanto a v0.3, porém maior e mais complexo || nearly as robust as v0.3, but larger and more complex
- `ufld_v07`: melhor em quadros limpos, mas o ganho não é temporal; o estado deriva em sequências longas || best on clean frames, but the gain is not temporal; its state drifts over long sequences
- `lite_baseline`: referência leve de um quadro || lightweight single-frame reference
- `lite_baseline_ct`: controle de treino da família lite || training control of the lite family
- `lite_v05`: **candidato embarcado**: usa o histórico sob oclusão || **embedded candidate**: uses its history under occlusion
- `lite_v05_static`: controle de capacidade: mesmas camadas, sem histórico; instável || capacity control: same layers, no history; unstable
- `lite_v06`: recorrente leve; ainda fraco || lightweight recurrent; still weak
