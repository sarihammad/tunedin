# TunedIn Music Recommender - Root Makefile
# This file provides convenient access to all project commands

.PHONY: help up down logs rebuild train export faiss seed clean test lint format pipeline

# Default target
help:
	@echo "TunedIn Music Recommender"
	@echo "========================"
	@echo ""
	@echo "Available commands:"
	@echo "  make help     - Show this help message"
	@echo "  make up       - Start all services with Docker Compose"
	@echo "  make down     - Stop all services"
	@echo "  make logs     - Show logs from all services"
	@echo "  make rebuild  - Rebuild and restart all services"
	@echo "  make train    - Train the LightGCN model"
	@echo "  make export   - Export user and item embeddings"
	@echo "  make faiss    - Build FAISS index from embeddings"
	@echo "  make seed     - Download sample Last.fm dataset"
	@echo "  make clean    - Clean up containers and volumes"
	@echo "  make test     - Run all tests"
	@echo "  make lint     - Lint all code"
	@echo "  make format   - Format all code"
	@echo "  make pipeline - Run full ML pipeline (seed -> train -> export -> faiss -> up)"
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make seed    # Download sample data"
	@echo "  2. make train   # Train the model"
	@echo "  3. make export  # Export embeddings"
	@echo "  4. make faiss   # Build FAISS index"
	@echo "  5. make up      # Start services"
	@echo ""
	@echo "Or run everything: make pipeline"

# Delegate to infra/Makefile
up down logs rebuild train export faiss seed clean test lint format pipeline:
	@$(MAKE) -C infra $@
