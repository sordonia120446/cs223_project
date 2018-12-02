.PHONY: all clean go wat

all: build

clean:
	@rm data/model.txt

build:
	@bash setup.sh

go:
	@python main.py

purge:
	@rm -rf venv/
