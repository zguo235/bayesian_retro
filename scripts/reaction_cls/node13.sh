#!/bin/bash
set -e
node=13
trap "echo Error from node$node" ERR

source /etc/profile.d/modules.sh
module load cuda/9.2
export PATH="$HOME/miniconda/bin:$PATH"
source $HOME/miniconda/bin/activate
conda activate python36

cd /groups1/gcc50461/reaction_cls/single_step/test$node
mkdir -p log

set +e

ga_gpu()
(
case $1 in
    -cuda?) cuda=${1:5:1}; shift;;
    -cuda) cuda=0; shift;;
    *) cuda=0;;
esac

OFFSET=$1

i=0
error_count=0
while [ "$i" -lt 5 ]; do
    REACTION=`expr $OFFSET + $i`
    start=`date +%s`
    SAVEFILE="reaction${REACTION}_`date +%Y%m%d-%H-%M-%S`_$RANDOM"
    date > log/$SAVEFILE.log
    python ga_cuda$cuda.py $REACTION $SAVEFILE >> log/$SAVEFILE.log
    exit_code=$?
    date >> log/$SAVEFILE.log
    end=`date +%s`
    runtime=`expr \( $end - $start \) / 60`
    if [ $exit_code -eq 0 ]; then
        echo "Experiment of reaction$REACTION on cuda$cuda finished at `date`. Elapsed time: $runtime minutes." >> output_error.log
        i=`expr "$i" + 1`
        error_count=0
    elif [ $exit_code -ne 0 -a $error_count -lt 10 ]; then
        echo "Error: Experiment of reaction$REACTION on cuda$cuda failed at `date`. Elapsed time: $runtime minutes. Restart this experiment." >> output_error.log
        error_count=`expr "$error_count" + 1`
    else
        echo "Error: Experiment of reaction$REACTION on cuda$cuda failed at `date`. Elapsed time: $runtime minutes. Jump to next reaction since 10 trials has done" >> output_error.log
        i=`expr "$i" + 1`
        error_count=0
    fi
done
)

ga_gpu -cuda0 260 &
pid[1]=$!

ga_gpu -cuda1 265 &
pid[2]=$!

ga_gpu -cuda2 270 &
pid[3]=$!

ga_gpu -cuda3 275 &
pid[4]=$!

top -bci -d 10 -w 180 -u acb11109zq >> top.log &
pid_top=$!
nvidia-smi -l 10 >> nvidia-smi.log &
pid_nvidia=$!

wait "${pid[@]}"
echo "Exit node $node with $?"
kill $pid_top
kill $pid_nvidia