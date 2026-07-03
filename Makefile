# Makefile for local development and deployment tasks
.PHONY: help build up down logs test lint format

help:
	@echo "Usage: make [target]"
	@echo "Targets: build, up, down, logs, format, lint"

build:
	docker build -f Dockerfile -t crisislens-api .
	docker build -f Dockerfile.streamlit -t crisislens-streamlit .

up:
	docker-compose up -d --build

down:
	docker-compose down

logs:
	docker-compose logs -f

format:
	black .
	isort .

lint:
	flake8 .
