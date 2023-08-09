PWD := $(shell pwd)
DIR := $(shell dirname $(PWD))
NAME := $(shell basename $(DIR))

EXP ?=

all:
	g++ main.cpp -g3 -Wall -Wno-unused-function -Wno-strict-aliasing -std=c++17 -o main -DNAME="$(NAME)" $(EXP) -pthread -lcrypto

clean:
	-rm -rf main