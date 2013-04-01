#!/bin/bash

JAVA_PATH=$JAVA_HOME/bin

deps=
for f in deps/*.jar; do
  deps=$f:$deps;
done;

$JAVA_PATH/jdb -sourcepath src -Xms2g -Xmx2g -cp $deps:bin $@

