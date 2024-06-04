for i in $(seq -w 00 10); do
    CUDA_VISIBLE_DEVICES=1 python generate_multiframe_v2.py -d /mnt/ssd2/jihun/dataset/sequences/$i -o /mnt/ssd2/jihun/dataset_MF/sequences/$i -n 4 -i 5
done
