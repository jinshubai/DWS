#!/bin/bash
./../../split.py ../../machinefile ../../../../data/rcv1
mpiexec -n 6 --machinefile ../../machinefile ./../../train -s 1 -c 1 /home/jing/dis_data/rcv1.sub ../../models/dtrain/l2svm_rcv1_6.sub.model
mpiexec -n 6 --machinefile ../../machinefile ./../../predict /home/jing/dis_data/rcv1.sub ../../models/dtrain/l2svm_rcv1_6.sub.model ../../outputs/dtrain/l2svm_rcv1_6.sub.out 
