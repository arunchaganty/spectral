#!/bin/bash

deps=
for f in deps/*.jar; do
  deps=$f:$deps;
done;

$JAVA_HOME/bin/jdb -sourcepath src -classpath $deps:bin $@

