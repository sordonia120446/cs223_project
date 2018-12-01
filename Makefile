.PHONY: all clean go wat

all: go

clean:
	@rm data/model.txt

go:
	@python main.py

wat:
	@python scripts/tutorial.py
