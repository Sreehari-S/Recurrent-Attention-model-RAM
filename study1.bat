python main.py --is_train=True --batch_size=20 ^
--patch_size=64 --num_patches=2 --device=0 ^
--conv=True --model=densenet121 --epoch=10 ^
--ckpt_dir=./ckpt/study1/densenet121_model ^
--data_dir="D:\Work\data\iprings\new_jpg_dataset" ^
--plot_dir=./plots/study1
echo densenet121 completed

python main.py --is_train=True --batch_size=20 ^
--patch_size=64 --num_patches=1 --device=0 ^
--conv=True --model=densenet169 --epoch=10 ^
--ckpt_dir=./ckpt/study1/densenet169_model ^
--data_dir="D:\Work\data\iprings\new_jpg_dataset" ^
--plot_dir=./plots/study1
echo densenet169 completed

python main.py --is_train=True --batch_size=20 ^
--patch_size=64 --num_patches=2 --device=0 ^
--conv=True --model=densenet169 --epoch=10 ^
--ckpt_dir=./ckpt/study1/densenet169_model ^
--data_dir="D:\Work\data\iprings\new_jpg_dataset" ^
--plot_dir=./plots/study1
echo densenet169 completed

rem python main.py --ckpt_dir=./ckpt/study1/densenet161_model --patch_size=32 --num_patches=1 --conv=True --model=densenet161 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
rem echo densenet161 completed
rem python main.py --ckpt_dir=./ckpt/study1/densenet161_model --patch_size=32 --num_patches=2 --conv=True --model=densenet161 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
rem echo densenet161 completed
rem python main.py --ckpt_dir=./ckpt/study1/densenet201_model --patch_size=32 --num_patches=1 --conv=True --model=densenet201 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
rem echo densenet201 completed
rem python main.py --ckpt_dir=./ckpt/study1/densenet201_model --patch_size=32 --num_patches=2 --conv=True --model=densenet201 --epoch=25 --device=0 --data_dir=./../data/slices --plot_dir=./plots/study1
rem echo densenet201 completed