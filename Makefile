# =========================
# Variabili di progetto
# =========================
PYTHON       = python3
VENV         = venv
PYTHON_VENV  = $(VENV)/bin/python3
PIP_VENV     = $(VENV)/bin/pip
MY_VENV		 = $(VENV)/bin/mypy

TRAIN_PROG   = train.py
PREDIC_PROG  = predict.py 
MAIN_SCRIPT  = train.py
REQS         = requirements.txt
INFO_TXT	 = info.txt

.PHONY: all install run run_train run_predict debug clean fclean lint lint-strict re

# =========================
# Target di default
# =========================
all: install
	@echo "0 0" > $(INFO_TXT)

	

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
run_train: install
	$(PYTHON_VENV) $(TRAIN_PROG)

run_predict: install
	$(PYTHON_VENV) $(PREDIC_PROG)

debug: install
	$(PYTHON_VENV) -m pdb $(MAIN_SCRIPT)

# =========================
# Pulizia
# =========================
clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache

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

