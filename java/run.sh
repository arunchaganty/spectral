#!/bin/bash

JAVA_PATH=$JAVA_HOME/bin

deps=
for f in deps/*.jar; do
  deps=$f:$deps;
done;

$JAVA_PATH/java -ea -Xms20g -Xmx25g -cp $deps:bin $@

