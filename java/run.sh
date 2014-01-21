#!/bin/bash

JAVA_PATH=$JAVA_HOME/bin

deps=
for f in deps/*.jar; do
  deps=$f:$deps;
done;

$JAVA_PATH/java -ea -Xms6g -Xmx16g -cp $deps:bin:test_data $@

