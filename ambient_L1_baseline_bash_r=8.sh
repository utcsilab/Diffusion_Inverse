for R in 2 4 6 8 
do
for sample in {0..100}
do
python inference_AmbientMRI_L1_baseline.py --gpu 3 --latent_seeds 10 11 12 13 14 --seed 10 --num_steps 500 --sample $sample --R $R --train_R 8
done
done