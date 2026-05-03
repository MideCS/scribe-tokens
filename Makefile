PYTHON := uv run

# --- Linting & Formatting ---
format:
	uv run ruff format

check:
	uv run ruff check --fix

format-check:
	make format
	make check

# --- Preprocessing ---
parse-iam:
	$(PYTHON) -m scripts.preprocess.parse_iam

parse-deepwriting:
	$(PYTHON) -m scripts.preprocess.parse_deepwriting

parse-didi:
	$(PYTHON) -m scripts.preprocess.parse_didi

split-deepwriting:
	$(PYTHON) -m scripts.preprocess.split_deepwriting

preprocess-iam: parse-iam
preprocess-deepwriting: parse-deepwriting split-deepwriting
preprocess-didi: parse-didi

# --- Tokenizer ---
train-tokenizers:
	$(PYTHON) -m scripts.train.tokenisers

eval-compression:
	$(PYTHON) -m scripts.eval.compression

eval-oov:
	$(PYTHON) -m scripts.eval.oov

eval-tokenizers: eval-compression eval-oov

# --- Training ---
train:
	$(PYTHON) -m scripts.train.main --all

train-test:
	$(PYTHON) -m scripts.train.main --all --test

train-parallel:
	bash scripts/train/parallel.sh

# --- Evaluating ---
eval:
	$(PYTHON) -m scripts.eval.main --all

eval-htr:
	$(PYTHON) -m scripts.eval.htr

eval-htg:
	$(PYTHON) -m scripts.eval.htg

# --- Plotting ---
plot:
	$(PYTHON) -m scripts.plot.compression
	$(PYTHON) -m scripts.plot.oov
	$(PYTHON) -m scripts.plot.discretization
	$(PYTHON) -m scripts.plot.double_descent
	$(PYTHON) -m scripts.plot.attention
	$(PYTHON) -m scripts.plot.convergence
	$(PYTHON) -m scripts.plot.results
	$(PYTHON) -m scripts.plot.scribe
	$(PYTHON) -m scripts.plot.htg

plot-compression:
	$(PYTHON) -m scripts.plot.compression

plot-oov:
	$(PYTHON) -m scripts.plot.oov

plot-discretization:
	$(PYTHON) -m scripts.plot.discretization

plot-double-descent:
	$(PYTHON) -m scripts.plot.double_descent

plot-attention:
	$(PYTHON) -m scripts.plot.attention

plot-convergence:
	$(PYTHON) -m scripts.plot.convergence

plot-results:
	$(PYTHON) -m scripts.plot.results

plot-scribe:
	$(PYTHON) -m scripts.plot.scribe

plot-htg:
	$(PYTHON) -m scripts.plot.htg

# --- Utilities ---
move-checkpoints:
	$(PYTHON) -m scripts.utils.move_best_checkpoint

fetch-metrics:
	$(PYTHON) -m scripts.utils.fetch_metrics

fetch-compute-time:
	$(PYTHON) -m scripts.utils.fetch_compute_time

kill:
	pgrep -f scribe-tokens | xargs kill -9

check-cuda:
	$(PYTHON) python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

tmux:
	tmux new-session -A -s train

-include local.mk
