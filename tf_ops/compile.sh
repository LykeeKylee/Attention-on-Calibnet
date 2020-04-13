#!/usr/bin/env bash

cd emd
sh tf*compile.sh
cd ..

cd grouping
sh tf*compile.sh
cd ..

cd interpolation
sh tf*compile.sh
cd ..

cd sampling
sh tf*compile.sh
cd ..
