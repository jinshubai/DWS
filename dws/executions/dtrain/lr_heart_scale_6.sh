#!/bin/bash
./../../split.py ../../machinefile ../../heart_scale
mpiexec -n 6 --machinefile ../../machinefile ./../../train -s 0 -c 1 /home/jing/dis_data/heart_scale.sub ../../models/dtrain/lr_heart_scale_6.sub.model
mpiexec -n 6 --machinefile ../../machinefile ./../../predict /home/jing/dis_data/heart_scale.sub ../../models/dtrain/lr_heart_scale_6.sub.model ../../outputs/dtrain/lr_heart_scale_6.sub.out 
