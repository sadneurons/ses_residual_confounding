
# Makefile for SES residual confounding project
# Usage:
#   make env        # show environment
#   make run        # execute notebook with papermill
#   make test       # run pytest tests
#   make html       # export executed notebook to HTML
#   make clean      # remove artifacts

NB=SES_residual_confounding_reproducible_v2.ipynb
OUT=SES_residual_confounding_executed.ipynb
HTML=SES_residual_confounding_executed.html

env:
	python -V
	pip list | grep -E 'numpy|pandas|statsmodels|patsy|matplotlib|papermill|nbconvert|pytest' || true

run:
	papermill $(NB) $(OUT) -p N_POP 200000 -p SEED 123 -p ART_DIR artifacts

test:
	pytest -q

html: run
	jupyter nbconvert --to html $(OUT) --output $(HTML)

clean:
	rm -rf artifacts __pycache__ *.html *.ipynb_checkpoint
