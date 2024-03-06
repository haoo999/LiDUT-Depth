# single image test
python test_simple.py --load_weights_folder path/to/your/weights/folder --image_path path/to/your/test/image

# multi image (folder) test
python test_simple.py --load_weights_folder path/to/your/weights/folder --image_path path/to/your/test/image_folder

# lidut-depth_640x192 mono training with IMAGENET pretrained encoder
python train.py --model_name test --data_path /path/to/your/kitti_data --mypretrain path/to/your/pretrained/weights --weights_init pretrained --num_workers 16 --num_epochs 30 --batch_size 12

# lidut-depth_1024x320 mono training with IMAGENET pretrained encoder
python train.py --model_name test --data_path /path/to/your/kitti_data --mypretrain path/to/your/pretrained/weights --weights_init pretrained --num_workers 16 --num_epochs 30 --batch_size 12 --width 1024 --height 320

# lidut-depth mono evaluating with IMAGENET pretrained encoder
python evaluate_depth.py --load_weights_folder path/to/your/weights/folder --data_path /path/to/your/kitti_data

# lidut-depth mono evaluating (all epochs together) with IMAGENET pretrained encoder
python evaluate_depth_all.py --load_weights_folder path/to/your/weights/folder --data_path /path/to/your/kitti_data --num_epochs number_of_your_training_epochs