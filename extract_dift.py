from dift.dift_dit import SDFeaturizer

import sys

print("开始执行脚本...")

import argparse
print("导入 argparse 成功。")

import decord
print("导入 decord 成功。")

decord.bridge.set_bridge("torch")
print("设置 decord bridge 为 torch。")

import torch
print("导入 torch 成功。")

# try:
#     from dift.dift_dit import SDFeaturizer
#     print("导入 dift.dift_dit.SDFeaturizer 成功。")
# except ImportError as e:
#     print(f"导入 dift.dift_dit.SDFeaturizer 失败: {e}")
#     sys.exit(1)

def main(args):
    dift = SDFeaturizer(args.model_id)

    try:
        video_reader = decord.VideoReader(args.input_path, height=480, width=720)
        print(f"读取视频成功，视频帧数: {len(video_reader)}")
    except Exception as e:
        print(f"读取视频失败: {e}")
        sys.exit(1)

    batch_ids = torch.linspace(0, len(video_reader)-1, 49).long()
    print(f"生成 batch_ids: {batch_ids}")

    try:
        frames = video_reader.get_batch(batch_ids)  # [f, h, w, c]
        print(f"获取帧成功，frames shape: {frames.shape}")
    except Exception as e:
        print(f"获取帧失败: {e}")
        sys.exit(1)

    frames = frames.permute(3, 0, 1, 2).unsqueeze(0)    # [b, c, f, h, w]
    video_tensor = (frames / 255.0 - 0.5) * 2
    print(f"处理后的视频张量 shape: {video_tensor.shape}")

    ft = dift.forward(
        video_tensor,
        prompt=args.prompt,
        t=args.t,
        return_layer_ids=args.return_layer_ids,
        ensemble_size=args.ensemble_size
    )
    print(f"特征提取成功，特征 shape: {ft.shape}")

    try:
        torch.save(ft.cpu(), args.output_path)  # save feature in the shape of [f, c, h, w]
        print(f"特征保存成功到 {args.output_path}")
    except Exception as e:
        print(f"保存特征失败: {e}")
        sys.exit(1)

parser = argparse.ArgumentParser(
    description='''Extract dift features from an input video and save them as a Torch tensor,
                in the shape of [f, c, h, w].''')

parser.add_argument('--model_id', default='THUDM/CogVideoX-5b', type=str, 
                    help='Model ID of the diffusion model in HuggingFace')
parser.add_argument('--t', default=250, type=int, 
                    help='Time step for diffusion, choose from range [0, 1000]')
parser.add_argument('--return_layer_ids', default=40, type=int,
                    help='Which upsampling block of U-Net to extract the feature map')
parser.add_argument('--prompt', default='', type=str,
                    help='Prompt used in the stable diffusion')
parser.add_argument('--ensemble_size', default=1, type=int, 
                    help='Number of repeated images in each batch used to get features')
parser.add_argument('--input_path', default="data/dance/videos/dance-jump.mp4", type=str,
                    help='Path to the input video file')
parser.add_argument('--output_path', type=str, default='dift.pt',
                    help='Path to save the output features as a Torch tensor')

args = parser.parse_args()

main(args)
