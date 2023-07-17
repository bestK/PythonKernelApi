.PHONY: build-docker run-dockerrun-docker remove-docker

build-docker:
	docker build -t python-kernel .

run-docker:
	docker rm -f python-kernel || true
	docker run -p 3333:3333 --name  python-kernel python-kernel


remove-docker:
	docker rm -f python-kernel || true
	
dev:
	poetry run dev

start:
	poetry run start