#!/bin/bash

JAVA_PATH=$JAVA_HOME/bin

deps=
for f in deps/*.jar; do
  deps=$f:$deps;
done;

$JAVA_PATH/java -Xms6g -Xmx6g -cp $deps:bin $@

