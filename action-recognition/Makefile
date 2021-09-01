.SILENT:
.PHONY: docs

COLOR_RESET = \033[0m
COLOR_GREEN = \033[32m
COLOR_YELLOW = \033[33m
PROJECT_NAME = `basename $(PWD)`
KERNEL_NAME := $(shell uname -s)

export GFLOW_DEBUG=true

.DEFAULT_GOAL = help

## Prints this help
help:
	printf "${COLOR_YELLOW}\n${PROJECT_NAME}\n\n${COLOR_RESET}"
	awk '/^[a-zA-Z\-\_0-9\.%]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "${COLOR_GREEN}$$ make %s${COLOR_RESET} %s\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)
	printf "\n"

setup:
	pip install -r requirements.txt

preprocess:
	python src/preprocess.py -i $(shell pwd)/input/data -o $(shell pwd)/output

train:
	python src/train.py -i $(shell pwd)/input/data/data.csv -o $(shell pwd)/output -e 100 -lr 1e-3 -bs 32 -d cpu:0

test:
	python src/test.py -m $(shell pwd)/output/model.pth -l $(shell pwd)/output/lb.pkl -i $(shell pwd)/input/videos/6096924-360p_487k_h264.mp4 -o $(shell pwd)/output -d cpu:0

test_dataflow:
	python -m src.dataflow_run \
	--path-model "gs://rec-alg/test/action-recognition/model.pth" \
	--label-bin "gs://rec-alg/test/action-recognition/lb.pkl" \
	--input "gs://rec-alg/datasets/videos_paths.csv" \
	--output "gs://rec-alg/test/action-recognition/output/results" \
	--device "cpu:0" \
	--max-num-workers 20 \
	--worker-machine-type "n2-standard-4" \
	--project-id "recomendacao-gcom" \