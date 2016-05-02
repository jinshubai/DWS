#!/bin/bash
./../../split.py ../../machinefile ../../../../data/rcv1
mpiexec -n 6 --machinefile ../../machinefile ./../../train -s 0 -c 1 /home/jing/dis_data/rcv1.sub ../../models/dtrain/lr_rcv1_6.sub.model
mpiexec -n 6 --machinefile ../../machinefile ./../../predict /home/jing/dis_data/rcv1.sub ../../models/dtrain/lr_rcv1_6.sub.model ../../outputs/dtrain/lr_rcv1_6.sub.out 
