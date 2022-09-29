#!/bin/sh

cp BrownianParticle_fifo.c tmp.c
magic=`date +%Y%m%d_%H%M%S`
cat tmp.c | sed 's/MAGIC_VATG[^"]*/MAGIC_VATG '$magic'/' > BrownianParticle_fifo.c
