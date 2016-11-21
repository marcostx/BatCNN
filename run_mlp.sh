
#!/bin/sh

TOOLS=../caffe/build/tools

GLOG_logtostderr=1 $TOOLS/caffe train -solver mlp_solver.prototxt | tee log01_mlp.txt
echo 'Done.'