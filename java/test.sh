#!/bin/bash

JAVA_PATH=$JAVA_HOME/bin
deps=
for f in deps/*.jar; do
  deps=$f:$deps;
done;

$JAVA_PATH/java -ea -cp $deps:bin org.junit.runner.JUnitCore $@


