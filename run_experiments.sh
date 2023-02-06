# sampling="RandomSampling"
sampling="LeastConfidence"

# datasets="cifar10lt svhnlt celebA"
datasets="cifar10lt"
n_epochs=50

# datasets="waterbirds"
# n_epochs=300

# algorithms="ERM RW groupDRO"
n_init_labeled=1000
n_queries=500
n_rounds=10

algorithms="RW groupDRO"
algorithms_for_sampling="ERM"

for dataset in $datasets; do 
for algorithm in $algorithms; do

cmd="python examples/run_expt.py --dataset ${dataset} --algorithm ${algorithm} "\
"--root_dir $PROJECT/fair_al_data "\
"--log_dir $SCRATCH/fair_al_logs/logs_${dataset}_${algorithm}_${sampling} --download "\
"--use_wandb --wandb_kwargs project=fair_al_nn "\
"entity=alext2 --wandb_api_key_path '/cluster/home/tifreaa/.wandb_api_key.txt' "\
"--n_queries=${n_queries} --n_rounds=${n_rounds} --n_epochs=${n_epochs} "\
"--n_init_labeled=${n_init_labeled} --strategy_name=${sampling}"

echo $cmd

sbatch --time 24:00:00 --ntasks 5 --mem-per-cpu 4096 --tmp=4000 --output slurm_${dataset}_${algorithm}_${sampling}.out --gpus=rtx_3090:1 --wrap "$cmd"

done
done
