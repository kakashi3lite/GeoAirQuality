# GeoAirQuality — shared developer/CI commands
# Single source of truth for lint/test/build so CI (GitLab + GitHub)
# and local development never drift apart.
#
# Usage:  make lint | make test | make test-integration | make build

PYTHON ?= python3
PIP    ?= pip3
REPORT ?= report.xml

.PHONY: install lint lint-fix test test-integration build scan

## Install dependencies for local development
install:
	cd api && $(PIP) install -r requirements.txt
	cd data-pipeline && $(PIP) install -r requirements.txt
	$(PIP) install pytest-cov bandit pip-audit

## Enforced lint (new code we own). BLOCKING in CI.
## NOTE: macOS external volumes (exFAT/FAT) create '._*' AppleDouble
## metadata files that break flake8/alembic — strip them defensively.
lint:
	find . -name '._*' -not -path './.git/*' -delete 2>/dev/null || true
	flake8 api/services/ tests/ --max-line-length=100
	black --check api/services/ tests/

## Auto-format our code
lint-fix:
	black api/services/ tests/

## Unit tests (no external services needed). BLOCKING in CI.
## Runs the supported suite (safety engine + API + news intelligence).
test:
	find . -name '._*' -not -path './.git/*' -delete 2>/dev/null || true
	cd api && $(PYTHON) -m pytest \
		../tests/test_safety_engine.py ../tests/test_safety_api.py \
		../tests/test_news.py \
		--cov=services --cov=main --cov-report=xml --cov-report=term \
		--junitxml=../$(REPORT) -q

## Legacy Dask pipeline tests — KNOWN-FAILING due to dask/pandas pin
## incompatibilities (dask is temporarily disabled in the project).
## Soft gate in CI (reported, non-blocking).
test-legacy:
	cd api && $(PYTHON) -m pytest ../tests/test_ingest.py -q

## Integration tests — require PostGIS + Redis (CI provides services).
test-integration:
	cd api && RUN_INTEGRATION=1 $(PYTHON) -m pytest ../tests/test_integration.py -v

## Build local images
build:
	docker build -f api/Dockerfile -t geoairquality/api:local api/
	docker build -f data-pipeline/Dockerfile -t geoairquality/pipeline:local data-pipeline/

## Static security scans (soft gate)
scan:
	bandit -r api/ -f json -o bandit.json -q || true
	pip-audit -r api/requirements.txt -r data-pipeline/requirements.txt || true
