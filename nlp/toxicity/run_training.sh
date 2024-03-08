model_name="xlm-roberta-large"
train_dir="results_"${model_name}
batch_size=128

docker run \
	--user=`id -u`:`id -g` \
	-e HOME=root \
	--privileged \
	--network=host \
	-ti \
	--gpus all \
	-v /sys:/sys \
	-v /shared2:/shared2 \
	-w `pwd` \
	-v /tmp:/tmp \
	--rm \
	-e CUDA_DEVICE_ORDER=PCI_BUS_ID \
	-e CUDA_VISIBLE_DEVICES="0,1,2,3" \
	dockerio.badoo.com/rnd/tensorflow/tensorflow:2.3.0-gpu \
		horovodrun \
			--output-filename ${train_dir}/horovod \
			-np 4 \
			-H localhost:4 \
			python3	\
				./train.py \
					--train_dir ${train_dir} \
					--dataset_dir /shared2/text_classification/datasets/multilingual_toxicity_kaggle_jigsaw_2020/ \
					--epochs_lr_update 20 \
					--min_eval_metric 0.01 \
					--print_per_train_steps 20  \
					--initial_learning_rate_transformer 2e-6 \
					--initial_learning_rate_head 2e-4 \
					--steps_per_train_epoch -1 \
					--steps_per_eval_epoch -1 \
					--model_name ${model_name} \
					--reset_on_lr_update \
					--use_good_checkpoint \
					--batch_size ${batch_size} \

