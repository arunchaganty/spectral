#!/bin/bash

SCALA_PATH=/user/angeli/programs/scala/bin

deps=
for f in deps/*.jar; do
  deps=$f:$deps;
done;

$SCALA_PATH/scala -cp $deps:bin/ $@

