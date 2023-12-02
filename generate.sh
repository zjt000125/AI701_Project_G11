CUDA_VISIBLE_DEVICES=0 python generate.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --test_data_dir='./demo/' \
  --output_dir="./outputs/"  \
  --suffix="demo" \
  --prompt="a photo of a S on the grass." \
  --sigma="0.6" \
  --domain_adaptor_path="./checkpoints/domain_adaptor_200000.pt" \
  --image_transformer_path="./checkpoints/image_transformer_100000.pt" \
  --seed=42
                                            
