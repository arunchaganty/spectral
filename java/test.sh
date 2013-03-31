#!/bin/bash

JAVA_PATH=/usr/bin
deps=
for f in deps/*.jar; do
  deps=$f:$deps;
done;

$JAVA_PATH/java -cp $deps:bin/ org.junit.runner.JUnitCore $@


