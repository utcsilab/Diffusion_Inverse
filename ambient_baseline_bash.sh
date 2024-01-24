for sample in {0..100}
do
python inference_AmbientMRI_baseline.py --gpu 3 --latent_seeds 10 11 12 13 14 15 --seed 10 --num_steps 500 --sample $sample --R 6&
python inference_AmbientMRI_baseline.py --gpu 4 --latent_seeds 10 11 12 13 14 15 --seed 10 --num_steps 500 --sample $sample --R 8

done
