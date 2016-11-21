
#!/bin/sh

TOOLS=../caffe/build/tools

GLOG_logtostderr=1 $TOOLS/caffe train -solver bat_solver.prototxt | tee log2.txt
echo 'Done.'
