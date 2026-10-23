PY ?= python3
NAME ?= external
.DEFAULT_GOAL := help

.PHONY: help data train eval export app api docs test sample external all clean

help:
	@echo "GlucoTwin"
	@echo "  make data     generate the synthetic EHR and 14-day CGM / wearable data"
	@echo "  make train    train the LightGBM models and the fusion network"
	@echo "  make eval     evaluate on the test patients, write results/ and plots"
	@echo "  make export   build the JSON bundle the web dashboard reads"
	@echo "  make app      start the clinician dashboard (React + Convex)"
	@echo "  make api      start the FastAPI reference service on :8000"
	@echo "  make docs     rebuild the architecture diagram, slides and screenshots"
	@echo "  make test     run the pytest suite"
	@echo "  make all      data, train, eval, export, test"

data:
	$(PY) src/ehr_gen.py
	$(PY) src/cgm_sim.py
	$(PY) src/make_sample.py

train:
	$(PY) src/train.py --trees
	$(PY) src/train.py --forecast
	$(PY) src/train.py --net

eval:
	$(PY) src/evaluate.py

export:
	$(PY) src/export_web.py

app:
	bun run dev

api:
	uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000

docs:
	$(PY) src/make_architecture.py
	$(PY) src/make_screenshots.py
	$(PY) src/make_slides.py

test:
	$(PY) -m pytest -q

# optional: score the CGM-only model on a real, openly licensed dataset
# download it yourself, then pass the CSV path
external:
	@test -n "$(CSV)" || (echo "usage: make external CSV=path/to/cgm.csv" && exit 1)
	$(PY) src/validate_external.py --input "$(CSV)" --name "$(NAME)"


all: data train eval export test
