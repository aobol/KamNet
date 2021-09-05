#!/bin/bash -l

#$$ -P snoplus

#$$ -l h_rt=${TIME}

#$$ -j y
#$$ -V
source /projectnb/snoplus/Mo_work_place/RAT/rat-pac/env.sh

python ${PROCESSOR} --input ${INPUT} --outputdir ${OUTPUT} --process_index ${PROCESSING_UNIT}
