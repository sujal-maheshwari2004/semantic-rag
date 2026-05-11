.PHONY: ingest benchmark test lint clean

ingest:
	uv run python -c "from data.corpus import CHUNKS; from src.pipeline import RAGPipeline; p = RAGPipeline(); p.ingest(CHUNKS); print('Ingest complete.')"

benchmark:
	uv run python -m benchmark.run

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/ benchmark/

clean:
	rm -rf .cache benchmark/results.json retrieval_benchmark.md