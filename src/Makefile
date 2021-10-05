.PHONY: clean lint test test-debug docker-build docker-shell docker-notebook \
	docker-ssh-server

PROJECT_NAME=gan_multimodal

## Note that certain goals in the makefile require certain environment
## variables to be set - see the body of the definitions for the goals.
## The following environment variables are used throught the Makefile:
## NOTEBOOK_PORT=8892, PROJECT_PATH=/home/itautkute/fashion-iq, SSH_PORT=7777

## Settings needed by docker {
USER_NAME=$(shell whoami)
UID_NUMBER=$(shell id -u)
GID_NUMBER=$(shell id -g)

## Information about build context that is passed to Docker image.
##
## We're going to use commit hash as a tag of the image. This allows for
## finding out which version of the code given Docker image comes from. 
## Note that this approach is not foolproof, because if you introduce an 
## uncommitted change in the local copy of the code, then the hash is going to 
## stay the same, i.e. it's going to point to the latest committed version. As
## a result, the hash is not going to correspond well to the code in the image.
COMMIT_HASH=$(shell git rev-parse HEAD)
CURR_DIR=$(shell pwd)
REPO_URL=$(shell git config --get remote.origin.url)
BUILD_INFO="repo=$(REPO_URL)\npwd=$(CURR_DIR)\ncommit=$(COMMIT_HASH)"

IMAGE_NAME=$(USER_NAME)_$(PROJECT_NAME)
CONTAINER_NAME=$(IMAGE_NAME)_default
## Uncomment one of the runtime options below
## Uncommment line below if your host supports Nvidia GPU (like Linux machine)
RUNTIME_OPTION=--runtime=nvidia
## Uncomment if your host doesn't support Nvidia GPU (like MacOSX)
#RUNTIME_OPTION=
## Settings needed by docker }

## Limit number of CPUs to be available to docker. 
## By default, docker can use all machine's CPUs.
CPUS_LIMIT=1

####################
## Normal commands
####################

clean:
	find . -name "*.pyc" -exec rm {} \;

lint:
	flake8 src

test:
	export PYTHONPATH=src; pytest

# Run tests but do not inhibit what's printed to stdout
test-debug:
	export PYTHONPATH=src; pytest -s

###################
## Docker
###################

## Build Docker image
docker-build:
	docker build \
		--build-arg USER_NAME=$(USER_NAME) \
		--build-arg UID_NUMBER=$(UID_NUMBER) \
		--build-arg GID_NUMBER=$(GID_NUMBER) \
		--build-arg BUILD_INFO=$(BUILD_INFO) \
		-t $(IMAGE_NAME) .

## Run the image and provide shell console.
docker-shell:
## Make sure that required environment variables are set
## Path to directory with data
ifndef PROJECT_PATH
	$(error PROJECT_PATH environment variable is not set)
endif
	docker run $(RUNTIME_OPTION) --rm -ti \
		-v $$PWD/:/home/$(USER_NAME)/project \
		-v $(PROJECT_PATH):/home/$(USER_NAME)/data \
		-e PROJECT_PATH=/home/$(USER_NAME)/data \
		--cpus=$(CPUS_LIMIT) \
		--name=$(CONTAINER_NAME) \
		$(IMAGE_NAME)

## Run the image and start a notebook server.
docker-notebook:
## Make sure that required environment variables are set
## Path to directory with data
ifndef PROJECT_PATH
	$(error PROJECT_PATH environment variable is not set)
endif
## Port for communication with Jupyter server
ifndef NOTEBOOK_PORT
	$(error NOTEBOOK_PORT environment variable  is not set)
endif
## Port for communication with SSH server
ifndef SSH_PORT
	$(error SSH_PORT environment variable  is not set)
endif
	docker run $(RUNTIME_OPTION) --rm -ti \
		-v $$PWD/:/home/$(USER_NAME)/project \
		-v $(PROJECT_PATH):/home/$(USER_NAME)/data \
		-e PROJECT_PATH=/home/$(USER_NAME)/data \
		--cpus=$(CPUS_LIMIT) \
		--name=$(CONTAINER_NAME) \
		-p $(SSH_PORT):22 \
		-p $(NOTEBOOK_PORT):$(NOTEBOOK_PORT) \
		$(IMAGE_NAME) \
		jupyter lab --ip 0.0.0.0 --port $(NOTEBOOK_PORT) \
			--no-browser --allow-root

## Run SSH server in a running container. 
## This is useful when you want to connect from PyCharm.
docker-ssh-server:
	docker exec -ti $(CONTAINER_NAME) sudo /usr/sbin/sshd -D
