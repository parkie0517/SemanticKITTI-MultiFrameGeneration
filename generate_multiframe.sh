for i in {01..10}; do
    CUDA_VISIBLE_DEVICES=1 python generate_multiframe.py -d /mnt/ssd2/jihun/dataset/sequences/$i -o /mnt/ssd2/jihun/dataset/multiframe/sequences/$i
done
