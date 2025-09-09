.PHONY: help up down logs rebuild train export faiss seed clean

# Default target
help:
	@echo "Available commands:"
	@echo "  up       - Start all services with Docker Compose"
	@echo "  down     - Stop all services"
	@echo "  logs     - Show logs from all services"
	@echo "  rebuild  - Rebuild and restart all services"
	@echo "  train    - Train the LightGCN model"
	@echo "  export   - Export user and item embeddings"
	@echo "  faiss    - Build FAISS index from embeddings"
	@echo "  seed     - Download sample Last.fm dataset"
	@echo "  clean    - Clean up containers and volumes"

# Docker Compose commands
up:
	@echo "Starting TunedIn services..."
	docker-compose -f ../docker/docker-compose.yml up -d
	@echo "Services started! Access:"
	@echo "  Frontend: http://localhost:5173"
	@echo "  Backend:  http://localhost:8000"
	@echo "  Grafana:  http://localhost:3001 (admin/admin)"
	@echo "  Prometheus: http://localhost:9090"

down:
	@echo "Stopping TunedIn services..."
	docker-compose -f ../docker/docker-compose.yml down

logs:
	docker-compose -f ../docker/docker-compose.yml logs -f

rebuild:
	@echo "Rebuilding and restarting services..."
	docker-compose -f ../docker/docker-compose.yml down
	docker-compose -f ../docker/docker-compose.yml build --no-cache
	docker-compose -f ../docker/docker-compose.yml up -d

# ML Pipeline commands
train:
	@echo "Training LightGCN model..."
	cd ../ml && python -m models.train

export:
	@echo "Exporting embeddings..."
	cd ../ml && python -m models.export_embeddings

faiss:
	@echo "Building FAISS index..."
	cd ../ml && python -m models.build_faiss

seed:
	@echo "Downloading sample Last.fm dataset..."
	cd ../ml && python -m data.scripts.lastfm_download

# Development commands
test:
	@echo "Running tests..."
	cd ../backend && python -m pytest tests/ -v
	cd ../frontend && npm test

lint:
	@echo "Linting code..."
	cd ../backend && ruff check . && black --check .
	cd ../frontend && npm run lint

format:
	@echo "Formatting code..."
	cd ../backend && black . && ruff check --fix .
	cd ../frontend && npm run format

# Cleanup
clean:
	@echo "Cleaning up..."
	docker-compose -f ../docker/docker-compose.yml down -v
	docker system prune -f
	rm -rf ../models/*.faiss ../models/*.npy ../models/*.pt
	rm -rf ../ml/runs/ ../ml/checkpoints/ ../ml/logs/

# Full pipeline
pipeline: seed train export faiss up
	@echo "Full pipeline completed!"
	@echo "Test the API: curl http://localhost:8000/rec/users/1?n=10"

