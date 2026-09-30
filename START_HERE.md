# Comece aqui / Start here

**TAC-UFLD: detecção de faixas com informação temporal.** Tiago Rodrigues, Eva Laussac, Everton Gomede (UTFPR, Cornélio Procópio).

## Português

Este pacote traz o código completo, a documentação e os resultados dos pilotos na base ELAS. As bases de dados e os pesos treinados não vêm no ZIP, por causa do tamanho.

1. **Ver os resultados sem instalar nada.** Abra `metrics_and_graphics/TAC-UFLD_resultados_pt.pdf`, ou `results_page/pt/index.html` no navegador. Os gráficos e as tabelas estão em `metrics_and_graphics/` e os relatórios de cada execução em `pilot_results/`. A mesma página está on-line em https://tiagorodrigues-gith.github.io/tac-ufld-lane-detection/pt/.
2. **Abrir na IDE.** No VS Code, use *Arquivo → Abrir Pasta* nesta pasta; as execuções prontas estão em *Executar e Depurar*. No PyCharm, use *File → Open* e configure o interpretador `.venv`. Os detalhes estão no `README.md`, seção "Opening the project in an IDE".
3. **Instalar**, com Python 3.10 ou mais novo e uma GPU NVIDIA para treinar:
   ```
   python -m venv .venv
   .venv\Scripts\activate            (Linux/macOS: source .venv/bin/activate)
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   pip install -e ".[all]"
   python -m tac_ufld doctor          # verifica o ambiente, a GPU e as bases
   ```
4. **Conferir o código.** `python -m pytest -q` (cerca de 10 a 40 min) e `python scripts/smoke_datasets.py` (todas as etapas em cada base, com cópias sintéticas quando a base não existe).
5. **Bases de dados.** Estão, em ordem: ELAS, CULane, TuSimple e OpenLane (esta última é uma das mais importantes para modelos temporais). Os links e o passo a passo estão no `README.md` ("Datasets: download and how to run them") e em `docs/DATASET_DOWNLOAD_GUIDE.md`.
6. **Experimento completo.** `python -m tac_ufld run --config configs/elas.yaml --confirm` leva cerca de 4 a 5 dias numa RTX 3050. O que rodar e o que decidir está em `docs/HANDOFF.md`.

## English

This package contains the full code, the documentation and the results of the ELAS pilots. Datasets and trained weights are not included, because of their size.

1. **See the results without installing anything:** `metrics_and_graphics/TAC-UFLD_results_en.pdf`, or `results_page/index.html` in a browser. The charts and tables are in `metrics_and_graphics/`, and every run's reports in `pilot_results/`. The same page is online at https://tiagorodrigues-gith.github.io/tac-ufld-lane-detection/.
2. **Open it in an IDE:** see `README.md` → "Opening the project in an IDE" (VS Code runs are ready in `.vscode/launch.json`).
3. **Install:** see the commands in step 3 above, then run `python -m tac_ufld doctor`.
4. **Check the code:** `python -m pytest -q` and `python scripts/smoke_datasets.py`.
5. **Datasets:** ELAS, then CULane, TuSimple and OpenLane (links and steps in `README.md` and `docs/DATASET_DOWNLOAD_GUIDE.md`).
6. **Full experiment:** `python -m tac_ufld run --config configs/elas.yaml --confirm`. What to run and what to decide is in `docs/HANDOFF.md`.
