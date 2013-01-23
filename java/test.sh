#!/bin/bash

JAVA_PATH=/usr/bin
deps=
for f in deps/*.jar; do
  deps=$f:$deps;
done;

$JAVA_PATH/java -cp $deps:bin/production/spectral/:bin/production/spectral-test/ org.junit.runner.JUnitCore $@


