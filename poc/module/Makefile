PWD := $(shell pwd)
DIR := $(shell dirname $(PWD))
NAME := $(shell basename $(DIR))

obj-m += powerdown.o 
powerdown-objs := module.o
flags := 

all: powerdown.ko

amd: flags += -DIS_AMD
amd: all


powerdown.ko: module.c interface.h 
	make -C /lib/modules/$(shell uname -r)/build EXTRA_CFLAGS="-DNAME=$(NAME) $(flags)" M=$(PWD) modules 

clean:
	make -C /lib/modules/$(shell uname -r)/build EXTRA_CFLAGS="-DNAME=$(NAME) $(flags)" M=$(PWD) clean

load: unload powerdown.ko
	-sudo insmod powerdown.ko

unload:
	-sudo rmmod powerdown.ko
