#!/bin/bash

JAVA_PATH=/usr/bin

deps=
for f in deps/*.jar; do
  deps=$f:$deps;
done;

$JAVA_PATH/java -Xms2g -Xmx2g -cp $deps:bin $@

