#!/bin/bash

JAVA_PATH=/usr/bin

deps=
for f in deps/*.jar; do
  deps=$f:$deps;
done;

$JAVA_PATH/java -ea -Xms2g -Xmx2g -cp $deps:bin/production/spectral/:bin/production/spectral-test $@

