#!/bin/bash

python3 -c "print('energy_unit_j=', 1.0 / 2**((0x`sudo rdmsr 0x606` & 0x1f00) >> 8))"

echo "Check if all cores are using the same TSC frequency:"
sudo turbostat -n 1  -show TSC_MHz 2>/dev/null

python3 -c "print('time_unit_s=', 1.0 / (`sudo turbostat -n 1  -show TSC_MHz 2>/dev/null | tail -n 1` * 10**6))"
