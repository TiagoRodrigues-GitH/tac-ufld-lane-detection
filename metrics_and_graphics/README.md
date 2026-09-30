# Métricas e gráficos / Metrics and graphics

**Autores:** Tiago Rodrigues, Eva Laussac, Everton Gomede (UTFPR, Cornélio Procópio)

Resultados dos pilotos do TAC-UFLD na base ELAS (30/09/2026). São resultados indicativos, com 2 a 3 sementes, treino curto e sem busca de hiperparâmetros. Só a execução completa (6 sementes, `configs/elas.yaml`) pode sustentar conclusões. Todos os números vêm das pastas `results/`; nada foi digitado à mão.

## Conteúdo

| Arquivo / pasta | O que é |
|---|---|
| `TAC-UFLD_resultados_pt.pdf` | A página de resultados completa, em português, em A4 (resumo, qual modelo comparar, resultados comentados, modelos, tabelas, ablações, latência, roteiro) |
| `TAC-UFLD_results_en.pdf` | A mesma página em inglês |
| `graphics/pt/`, `graphics/en/` | Os 8 gráficos principais em PNG e SVG, em português e em inglês |
| `graphics/report_figures/` | Figuras geradas automaticamente pelo relatório de cada execução |
| `metrics/*.csv` | Todas as tabelas de métricas (uma linha por modelo e semente, ou média ± desvio) |
| `metrics/all_metrics.xlsx` | As mesmas tabelas numa única planilha, uma aba por tabela |

## Os gráficos

| Gráfico | Mostra |
|---|---|
| `01_augmentation_ablation` | O que corrigiu o overfitting da UFLD: só o aumento de dados geométrico (0,858 de lane F1 no teste, contra 0,068 da receita original) |
| `02_clean_heldout_f1` | Lane F1 de cada modelo nas cenas de teste nunca vistas, com quadros limpos |
| `03_occluded_current_frame` | Cada modelo com o quadro atual limpo e ocluído (histórico limpo): os modelos temporais UFLD v0.3 e v0.6 e o lite v0.5 quase não perdem nada |
| `04_degradation_per_type` | Lane F1 da família UFLD por tipo de degradação do quadro atual (oclusão, desfoque, escurecimento, ruído) |
| `05_accuracy_vs_parameters` | Acurácia × tamanho: o lite v0.5 chega a 0,850 com 2,8 M de parâmetros, contra 0,881 da UFLD com 21,8 M |
| `06_history_length` | Ganho da UFLD v0.4 sobre a baseline para cada comprimento e espaçamento do histórico |
| `07_kalman_jitter` | Oscilação (jitter) de cada modelo com e sem o filtro de Kalman na saída |
| `08_latency_rtx3050` | Tempo por quadro na RTX 3050, separado em pré-processamento, modelo e pós-processamento |

## Tabelas principais

- `pilot2_heldout_mean_std.csv`: piloto 2 (11 modelos, 2 sementes, até 4 épocas), todas as métricas nas cenas de teste.
- `lite_long_heldout_mean_std.csv`: família lite com até 16 épocas, incluindo o controle de capacidade (lite v0.5 static).
- `*_robustness_summary.csv`: lane F1 com o quadro atual degradado (por tipo) e o F1 com quadros limpos.
- `*_paired_tests.csv`: comparações pareadas por semente (Wilcoxon + Holm); com 2 sementes todas são "underpowered" (poucas sementes).
- `ablation_augmentation_*.csv`, `ablation_history_results.csv`: as duas ablações.
- `*_efficiency.csv`, `latency_desktop_rtx3050.csv`: parâmetros, GMACs, latência e memória.

## Como regenerar

```
python scripts/build_metrics_folder.py                  # tabelas e gráficos
python -m tac_ufld site --lang pt --out site/pt --runs results/elas_pilot_v2 results/elas_lite_long --notes docs/PILOT_V2_FINDINGS.md --authors "Tiago Rodrigues" "Eva Laussac" "Everton Gomede" --affiliation "UTFPR, Cornélio Procópio, Brazil"
```

Depois, abra `site/pt/index.html` no Chrome ou no Edge e use Imprimir → Salvar como PDF. A página tem estilos próprios para impressão em A4.

---

**English summary.** Pilot results of TAC-UFLD on ELAS (30 Sep 2026; 2–3 seeds, short training, no hyper-parameter search, so indicative only). The folder contains the full results page as PDF (Portuguese and English), the eight main charts (PNG and SVG, both languages), the reports' own figures, every metrics table as CSV and one Excel workbook. Rebuild it with `python scripts/build_metrics_folder.py`; all numbers are read from `results/`.
