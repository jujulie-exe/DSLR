# =========================
# Variabili di progetto
# =========================
PYTHON       = python3
VENV         = venv
PYTHON_VENV  = $(VENV)/bin/python3
PIP_VENV     = $(VENV)/bin/pip
MY_VENV		 = $(VENV)/bin/mypy

TRAIN_PROG   = logreg_train.py
PREDIC_PROG  = logreg_predict.py
DESCRIBE_PROG = describe.py
HISTOGRAM_PROG = histogram.py
PAIR_PLOT_PROG = pair_plot.py
SCATTER_PLOT_PROG = scatter_plot.py

MAIN_SCRIPT  = train.py
REQS         = requirements.txt
WEIGHTS_FILE = weight.csv
TEST_FILE    = datasets/dataset_test.csv

.PHONY: all install run run_train run_predict run_describe run_histogram run_pair_plot run_scatter_plot debug clean fclean lint lint-strict re

# =========================
# Target di default
# =========================
all: install
	@echo "" > $(WEIGHTS_FILE)



	

# =========================
# Creazione del virtual environment se non esiste
# =========================
$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment creato."

# =========================
# Installazione dipendenze
# =========================
$(VENV)/.installed: $(VENV)/bin/activate requirements.txt
	$(PIP_VENV) install --upgrade pip
	$(PIP_VENV) install -r $(REQS)
	touch $(VENV)/.installed

install: $(VENV)/.installed


# =========================
# Esecuzione script
# =========================
run_describe: install
	$(PYTHON_VENV) $(DESCRIBE_PROG)

run_histogram: install
	$(PYTHON_VENV) $(HISTOGRAM_PROG)
	
run_pair_plot: install
	$(PYTHON_VENV) $(PAIR_PLOT_PROG)

run_scatter_plot: install
	$(PYTHON_VENV) $(SCATTER_PLOT_PROG)

run_train: install
	$(PYTHON_VENV) $(TRAIN_PROG)

run_predict: install
	$(PYTHON_VENV) $(PREDIC_PROG) $(TEST_FILE) $(WEIGHTS_FILE)

debug: install
	$(PYTHON_VENV) -m pdb $(MAIN_SCRIPT)

# =========================
# Pulizia
# =========================
clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache $(WEIGHTS_FILE)

fclean: clean
	rm -rf $(VENV)

re: fclean all

# =========================
# Linting
# =========================
lint:
	$(MY_VENV) . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	$(MY_VENV) . --strict

