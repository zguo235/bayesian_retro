#!/bin/bash
set -e
trap "echo Error from quick start script" ERR

source /etc/profile.d/modules.sh
# Next line is an example for loading cuda environment. Comment out next line if cuda is already in your path.
module load cuda/9.2
export PATH="$HOME/miniconda/bin:$PATH"
source $HOME/miniconda3/bin/activate
conda activate python36

cd ./experiments_single-step/test0
mkdir -p log

set +e

ga_gpu()
(

OFFSET=$1

i=0
while [ "$i" -lt 1 ]; do
    REACTION=`expr $OFFSET + $i`
    start=`date +%s`
    SAVEFILE="reaction${REACTION}_`date +%Y%m%d-%H-%M-%S`_$RANDOM"
    date > log/$SAVEFILE.log
    python ga_retrosynthesis.py $REACTION $SAVEFILE >> log/$SAVEFILE.log
    exit_code=$?
    date >> log/$SAVEFILE.log
    end=`date +%s`
    runtime=`expr \( $end - $start \) / 60`
    if [ $exit_code -eq 0 ]; then
        echo "Experiment of reaction$REACTION finished at `date`. Elapsed time: $runtime minutes." >> output_error.log
        i=`expr "$i" + 1`
    else
        echo "Error: Experiment of reaction$REACTION failed at `date`. Elapsed time: $runtime minutes. Restart this experiment." >> output_error.log
    fi
done
)

ga_gpu 0 &
pid[1]=$!

top -bci -d 10 -w 180 >> top.log &
pid_top=$!
nvidia-smi -l 10 >> nvidia-smi.log &
pid_nvidia=$!

wait "${pid[@]}"
echo "Exit quick start script with exit code $?"
kill $pid_top
kill $pid_nvidia
