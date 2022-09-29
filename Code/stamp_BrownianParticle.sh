#!/bin/sh

cp BrownianParticle_magic.h tmp.h
magic=`date +%Y%m%d_%H%M%S`
cat tmp.h | sed 's/MAGIC_VATG[^"]*/MAGIC_VATG '$HOST':'$magic'/' > BrownianParticle_magic.h
