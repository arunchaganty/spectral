#!/bin/bash
java7 -cp deps/ejml-0.20.jar:deps/javatuples-1.2.jar:deps/fig.jar:bin learning.spectral.applications.SentenceHMMLearner $@
