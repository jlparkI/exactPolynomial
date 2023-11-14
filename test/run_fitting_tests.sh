#!/bin/bash

#Run fitting tests
cd fitting_tests
python test_lbfgs_fit.py
python test_ista_fit.py
cd ..
