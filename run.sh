rm -rf plot
rm -rf ./checkpoints
python gen_gcd.py
python gpt.py --emb_dim=256 \
    --nheads=8 \
    --batch_size=4096 \
    --num_layers=16 \
    --num_epochs=900 \
    --lr=5e-5 \
    --save_every=100
