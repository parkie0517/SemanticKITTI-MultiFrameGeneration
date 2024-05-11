for i in {00..10}; do
    CUDA_VISIBLE_DEVICES=1 python generate_multiframe.py -d /mnt/ssd2/jihun/dataset/sequences/$i -o /mnt/ssd2/jihun/dataset/multiframe/sequences/$i -n 4 -i 5
done
