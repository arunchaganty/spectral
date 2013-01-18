#!/bin/bash

JAVA_PATH=/usr/bin

deps=
for f in deps/*.jar; do
  deps=$f:$deps;
done;

$JAVA_PATH/jdb -sourcepath src -classpath $deps:bin $@

