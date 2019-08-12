python3 main.py --is_train=False --is_plot=True --mode='test' --num_plots=12 --plot_name="plots" \
--ckpt_dir=./ckpt/study1/densenet201_model --patch_size=32 --num_patches=2 \
--conv=True --model=densenet201 --device=0 --data_dir=./../data/slices --best=True --plot_dir=./plots/
python3 plot_glimpses.py --plot_dir=./plots/densenet201_6_32x32_2_2/ --epoch=0 --name="plots"
echo Hello
