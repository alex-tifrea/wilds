TAGS="[\'main\']"

samplings="RandomSampling LeastConfidence"
# samplings="RandomSampling"
# samplings="LeastConfidence"

default_algorithm_for_sampling=""
algorithms="RW groupDRO"
n_init_labeled=1000
n_queries=500
n_rounds=10

datasets="cifar10lt svhnlt celebA"
# datasets="cifar10lt svhnlt"
n_epochs=50

# datasets="waterbirds"
# n_epochs=250

datasets="inaturalist"
n_epochs=20
n_init_labeled=10000
n_queries=5000

# samplings="LeastConfidence"
# algorithms="RW groupDRO"
# default_algorithm_for_sampling="ERM"

for dataset in $datasets; do 
for sampling in $samplings; do
for algorithm in $algorithms; do

if [ -z $default_algorithm_for_sampling ]; then
  algorithm_for_sampling=$algorithm
else
  algorithm_for_sampling=$default_algorithm_for_sampling
fi

timestamp=$(date +%d%m%Y:%H%M%S)

cmd="python examples/run_expt.py --dataset ${dataset} --algorithm ${algorithm} "\
"--algorithm_for_sampling $algorithm_for_sampling "\
"--root_dir $PROJECT/fair_al_data "\
"--log_dir $SCRATCH/fair_al_logs/logs_${dataset}_${algorithm_for_sampling}_${algorithm}_${sampling}_${timestamp} --download "\
"--use_wandb --wandb_kwargs project=fair_al_nn "\
"entity=alext2 tags=${TAGS} --wandb_api_key_path '/cluster/home/tifreaa/.wandb_api_key.txt' "\
"--n_queries=${n_queries} --n_rounds=${n_rounds} --n_epochs=${n_epochs} "\
"--n_init_labeled=${n_init_labeled} --strategy_name=${sampling}"

echo $cmd

sbatch --time 24:00:00 --ntasks 5 --mem-per-cpu 4096 --tmp=4000 --output slurm_${dataset}_${algorithm_for_sampling}_${algorithm}_${sampling}_${timestamp}.out --gpus=rtx_3090:1 --wrap "$cmd" 

done
done
done
