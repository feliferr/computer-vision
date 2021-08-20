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

train:
	python src/train.py -i $(shell pwd)/input/data/data.csv -o $(shell pwd)/output -e 100 -lr 1e-3 -bs 32 -d cpu:0
