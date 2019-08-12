# running on bmi
# export CUDA_VISIBLE_DEVICES=0
# python main.py --ckpt_dir=./ckpt/study1/resnet18_model --patch_size=32 --num_patches=1 --conv=True --model=resnet18 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
# echo resnet18 completed
# python main.py --ckpt_dir=./ckpt/study1/resnet18_model --patch_size=32 --num_patches=2 --conv=True --model=resnet18 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1 
# echo resnet18 completed
# python main.py --ckpt_dir=./ckpt/study1/resnet34_model --patch_size=32 --num_patches=1 --conv=True --model=resnet34 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
# echo resnet34 completed
# python main.py --ckpt_dir=./ckpt/study1/resnet34_model --patch_size=32 --num_patches=2 --conv=True --model=resnet34 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
# echo resnet34 completed
# python main.py --ckpt_dir=./ckpt/study1_repeat/resnet50_model --patch_size=32 --num_patches=1 --conv=True --model=resnet50 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1_repeat
# echo resnet50 completed
# python main.py --ckpt_dir=./ckpt/study1_repeat/resnet50_model --patch_size=32 --num_patches=2 --conv=True --model=resnet50 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1_repeat
# echo resnet50 completed
# python main.py --ckpt_dir=./ckpt/study1/resnet101_model --patch_size=32 --num_patches=1 --conv=True --model=resnet101 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
# echo resnet101 completed
# python main.py --ckpt_dir=./ckpt/study1/resnet101_model --patch_size=32 --num_patches=2 --conv=True --model=resnet101 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
# echo resnet101 completed
# python main.py --ckpt_dir=./ckpt/study1/resnet152_model --patch_size=32 --num_patches=1 --conv=True --model=resnet152 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
# echo resnet152 completed
# python main.py --ckpt_dir=./ckpt/study1/resnet152_model --patch_size=32 --num_patches=2 --conv=True --model=resnet152 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
# echo resnet152 completed
# python main.py --ckpt_dir=./ckpt/study1/densenet121_model --patch_size=32 --num_patches=1 --conv=True --model=densenet121 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
# echo densenet121 completed
python main.py --ckpt_dir=./ckpt/study1/densenet121_model --patch_size=32 --num_patches=2 --conv=True --model=densenet121 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
echo densenet121 completed
python main.py --ckpt_dir=./ckpt/study1/densenet169_model --patch_size=32 --num_patches=1 --conv=True --model=densenet169 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
echo densenet169 completed
python main.py --ckpt_dir=./ckpt/study1/densenet169_model --patch_size=32 --num_patches=2 --conv=True --model=densenet169 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
echo densenet169 completed
python main.py --ckpt_dir=./ckpt/study1/densenet161_model --patch_size=32 --num_patches=1 --conv=True --model=densenet161 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
echo densenet161 completed
python main.py --ckpt_dir=./ckpt/study1/densenet161_model --patch_size=32 --num_patches=2 --conv=True --model=densenet161 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
echo densenet161 completed
python main.py --ckpt_dir=./ckpt/study1/densenet201_model --patch_size=32 --num_patches=1 --conv=True --model=densenet201 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
echo densenet201 completed
python main.py --ckpt_dir=./ckpt/study1/densenet201_model --patch_size=32 --num_patches=2 --conv=True --model=densenet201 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
echo densenet201 completed




# echo Abalation Study 2
#
# echo ResNet50 Scale 1 Patch Size 32
# python main.py --ckpt_dir=./ABALATION_STUDY_2/resnet50_model --patch_size=32 --num_patches=1 --conv=True --model=resnet50 --epoch=1 --device=0 --data_dir=./../data/slices
# echo ResNet50 Scale 2 Patch Size 32
# python main.py --ckpt_dir=./ABALATION_STUDY_2/resnet50_model --patch_size=32 --num_patches=2 --conv=True --model=resnet50 --epoch=1 --device=0 --data_dir=./../data/slices

# echo ResNet50 Scale 1 Patch Size 64
# python main.py --ckpt_dir=./ABALATION_STUDY_2/resnet50_model --patch_size=64 --num_patches=1 --conv=True --model=resnet50 --epoch=1 --device=0 --data_dir=./../data/slices
# echo ResNet50 Scale 2 Patch Size 64
# python main.py --ckpt_dir=./ABALATION_STUDY_2/resnet50_model --patch_size=64 --num_patches=2 --conv=True --model=resnet50 --epoch=1 --device=0 --data_dir=./../data/slices


# echo ResNet50 Scale 1 Patch Size 96
# python main.py --ckpt_dir=./ABALATION_STUDY_2/resnet50_model --patch_size=96 --num_patches=1 --conv=True --model=resnet50 --epoch=1 --device=0 --data_dir=./../data/slices
#echo ResNet50 Scale 2 Patch Size 64
#python main.py --ckpt_dir=./ABALATION_STUDY_2/resnet50_model --patch_size=64 --num_patches=2 --conv=True --model=resnet50 --epoch=25 --device=0 --data_dir=./../data/slices
