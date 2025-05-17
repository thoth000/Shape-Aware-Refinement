import os
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from models.unet import UNet as Model
import dataloader.drive_loader as drive
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def check_args():
    parser = argparse.ArgumentParser()
    
    # 基本パラメータ
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--result_dir', type=str, default='result/sample')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--dataset', type=str, default='drive', choices=['pascal', 'pascal-sbd', 'davis2016', 'cityscapes-processed', 'drive'])
    parser.add_argument('--transform', type=str, default='standard', choices=['fr_unet', 'standard'])
    parser.add_argument('--pretrained_path', type=str, default='/home/sano/documents/swin_unet_drive/models/swin_tiny_patch4_window7_224.pth')
    
    # モデル固有のパラメータ
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--feature_scale', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--fuse', type=bool, default=True)
    parser.add_argument('--out_ave', type=bool, default=True)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='elu')
    
    # 異方性拡散パラメータ
    parser.add_argument('--num_iterations', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--save_interval', type=int, default=10, help='拡散プロセスの結果を保存する間隔')
    
    # データセットパラメータ
    parser.add_argument('--dataset_path', type=str, default="/home/sano/dataset/DRIVE")
    parser.add_argument('--dataset_opt', type=str, default="pro")
    parser.add_argument('--image_index', type=int, default=0, help='テストセットから処理する画像のインデックス')
    
    args = parser.parse_args()
    
    return args

def gradient(scalar_field):
    """
    スカラー場の勾配を計算。

    Parameters:
        scalar_field (torch.Tensor): スカラー場（形状: [B, 1, H, W]）。

    Returns:
        grad_x (torch.Tensor): x方向の勾配（形状: [B, 1, H, W]）。
        grad_y (torch.Tensor): y方向の勾配（形状: [B, 1, H, W]）。
    """
    # 有限差分係数（精度8次）
    coeff = torch.tensor([-1/280, 4/105, -1/5, 4/5, 0, -4/5, 1/5, -4/105, 1/280],
                         dtype=scalar_field.dtype, device=scalar_field.device)

    # x方向の勾配計算
    x_pad = F.pad(scalar_field, (4, 4, 0, 0), mode='replicate')
    grad_x = sum(coeff[i] * x_pad[..., i:i+scalar_field.size(-1)] for i in range(9))

    # y方向の勾配計算
    y_pad = F.pad(scalar_field, (0, 0, 4, 4), mode='replicate')
    grad_y = sum(coeff[i] * y_pad[..., i:i+scalar_field.size(-2), :] for i in range(9))

    return grad_x, grad_y

def divergence(grad_x, grad_y):
    """
    ベクトル場の発散を計算。

    Parameters:
        grad_x (torch.Tensor): x方向のベクトル場（形状: [B, 1, H, W]）。
        grad_y (torch.Tensor): y方向のベクトル場（形状: [B, 1, H, W]）。

    Returns:
        divergence (torch.Tensor): 発散（形状: [B, 1, H, W]）。
    """
    # 有限差分係数（精度8次）
    coeff = torch.tensor([-1/280, 4/105, -1/5, 4/5, 0, -4/5, 1/5, -4/105, 1/280],
                         dtype=grad_x.dtype, device=grad_x.device)

    # x方向の発散計算
    dx_pad = F.pad(grad_x, (4, 4, 0, 0), mode='replicate')
    div_x = sum(coeff[i] * dx_pad[..., i:i+grad_x.size(-1)] for i in range(9))

    # y方向の発散計算
    dy_pad = F.pad(grad_y, (0, 0, 4, 4), mode='replicate')
    div_y = sum(coeff[i] * dy_pad[..., i:i+grad_y.size(-2), :] for i in range(9))

    return div_x + div_y

def create_anisotropic_tensor_from_vector(vectors, lambda1=1.0, lambda2=0.0):
    """
    方向ベクトルに基づく異方性拡散テンソルを生成。

    Parameters:
        vectors (torch.Tensor): 方向ベクトル（形状: [B, 2, H, W]）。
        lambda1 (float): 主方向の拡散強度。
        lambda2 (float): 直交方向の拡散強度。

    Returns:
        D (torch.Tensor): 異方性拡散テンソル（形状: [B, 2, 2, H, W]）。
    """
    B, _, H, W = vectors.shape

    # ベクトルを正規化
    v_x = vectors[:, 0].unsqueeze(1)  # [B, 1, H, W]
    v_y = vectors[:, 1].unsqueeze(1)  # [B, 1, H, W]
    
    # テンソル構築
    vvT = torch.zeros(B, 2, 2, H, W, device=vectors.device, dtype=vectors.dtype)  # [B, 2, 2, H, W]
    vvT[:, 0, 0] = (v_x * v_x).squeeze(1)  # [B, H, W]
    vvT[:, 0, 1] = (v_x * v_y).squeeze(1)  # [B, H, W]
    vvT[:, 1, 0] = (v_y * v_x).squeeze(1)  # [B, H, W]
    vvT[:, 1, 1] = (v_y * v_y).squeeze(1)  # [B, H, W]

    I = torch.eye(2, device=vectors.device).view(1, 2, 2, 1, 1)  # 単位行列
    D = lambda1 * vvT + lambda2 * (I - vvT)

    return D

def compute_divergence(preds, diffusion_tensor):
    """
    拡散テンソルに基づく発散を計算。
    
    Parameters:
        preds (torch.Tensor): スカラー場（形状: [B, 1, H, W]）。
        diffusion_tensor (torch.Tensor): 拡散テンソル（形状: [B, 2, 2, H, W]）。
        
    Returns:
        div (torch.Tensor): 発散（形状: [B, 1, H, W]）。
    """
    # 勾配計算
    grad_x, grad_y = gradient(preds)
    grad = torch.cat([grad_x, grad_y], dim=1)
    # 異方性テンソル変換
    tg = torch.einsum('bijhw,bjhw->bihw', diffusion_tensor, grad)
    # ダイバージェンス
    div_x, div_y = tg[:,0], tg[:,1]
    div = divergence(div_x.unsqueeze(1), div_y.unsqueeze(1))
    return F.relu(div)  # [B,1,H,W]

def anisotropic_diffusion(preds, diffusion_tensor, num_iterations=100, gamma=0.1, save_dir=None, save_interval=10):
    """
    異方性拡散プロセスを実行し、各イテレーションの結果を保存。

    Parameters:
        preds (torch.Tensor): スカラー場（形状: [B, 1, H, W]）。
        diffusion_tensor (torch.Tensor): 異方性拡散テンソル（形状: [B, 2, 2, H, W]）。
        num_iterations (int): 拡散の反復回数。
        gamma (float): 拡散の強さ。
        save_dir (str): 結果を保存するディレクトリ。
        save_interval (int): 結果を保存する間隔。

    Returns:
        preds (torch.Tensor): 拡散後のスカラー場（形状: [B, 1, H, W]）。
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 初期予測を保存
    if save_dir:
        save_tensor_as_image(preds, os.path.join(save_dir, f'prob_iter_0.png'))
        save_tensor_as_image((preds > 0.5).float(), os.path.join(save_dir, f'mask_iter_0.png'))
    
    for iter_idx in range(1, num_iterations + 1):
        # RK4法による拡散
        h = gamma
        k1 = compute_divergence(preds, diffusion_tensor)
        k2 = compute_divergence(preds + 0.5*h*k1, diffusion_tensor)
        k3 = compute_divergence(preds + 0.5*h*k2, diffusion_tensor)
        k4 = compute_divergence(preds + h*k3, diffusion_tensor)

        # 更新
        preds = preds + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        preds = preds.clamp(0, 1)
        
        # 指定した間隔で結果を保存
        if save_dir and (iter_idx % save_interval == 0 or iter_idx == num_iterations):
            save_tensor_as_image(preds, os.path.join(save_dir, f'prob_iter_{iter_idx}.png'))
            save_tensor_as_image((preds > 0.5).float(), os.path.join(save_dir, f'mask_iter_{iter_idx}.png'))
    
    return preds

def save_tensor_as_image(tensor, filepath):
    """
    テンソルを画像として保存。
    
    Parameters:
        tensor (torch.Tensor): 保存するテンソル（形状: [B, 1, H, W]）。
        filepath (str): 保存先のファイルパス。
    """
    # テンソルをCPUに移動し、NumPy配列に変換
    img_np = tensor.squeeze().cpu().numpy()
    # 値を0-255の範囲に正規化
    img_np = (img_np * 255).astype(np.uint8)
    # PILイメージに変換して保存
    Image.fromarray(img_np).save(filepath)

def save_main_out_image(output_tensor, filepath, cmap="viridis"):
    """
    output_tensorをカラーマップで画像として保存する関数
    :param output_tensor: 保存するテンソル
    :param filepath: 画像の保存先のパス
    :param cmap: 使用するカラーマップ（デフォルトは 'viridis'）
    """
    # output_tensorをnumpy配列に変換
    output_numpy = output_tensor.squeeze().cpu().detach().numpy()

    # カラーマップを使って画像として保存
    plt.imshow(output_numpy, cmap=cmap)
    plt.colorbar()
    plt.savefig(filepath)  # ファイルパスに画像を保存
    plt.close()  # メモリを解放するためにプロットを閉じる

def save_masked_output_with_imsave(output_tensor, mask_tensor, filepath, normalize=True, cmap="viridis", vmin=-1, vmax=1):
    """
    真値マスクで指定された部分だけをカラーマップで可視化して保存する関数（plt.imsaveを使用）。
    
    Args:
        output_tensor (torch.Tensor): 可視化したい出力テンソル（shape: [1, 1, H, W]）。
        mask_tensor (torch.Tensor): 01バイナリの真値マスクテンソル [1, 1, H, W]。
        filepath (str): 保存するファイルパス。
        normalize (bool): 出力テンソルを [0, 1] に正規化するかどうか。
        cmap (str): 使用するカラーマップ（デフォルトは 'viridis'）。
    """
    # 出力テンソルとマスクテンソルを NumPy 配列に変換（[H, W] に変換）
    output_numpy = output_tensor[0, 0].cpu().detach().numpy()
    mask_numpy = mask_tensor[0, 0].cpu().detach().numpy()

    # マスクを適用して出力テンソルをフィルタリング
    masked_output = np.where(mask_numpy > 0.5, output_numpy, np.nan)

    # `np.nan` をカラーマップの白色に置き換え
    masked_output = np.ma.masked_invalid(masked_output)  # NaNをマスク
    cmap_instance = plt.cm.get_cmap(cmap)  # カラーマップを取得
    cmap_instance.set_bad(color='white')  # マスクされた部分を白色に設定

    # `plt.imsave` を使用して保存
    plt.imsave(filepath, masked_output, cmap=cmap_instance, vmin=vmin, vmax=vmax)

def save_direction_vector_as_image(vector, filepath):
    """
    方向ベクトルをカラーマップで可視化して保存。
    evaluate.pyと同じ方法でベクトル場を可視化する。
    
    Parameters:
        vector (torch.Tensor): 方向ベクトル（形状: [B, 2, H, W]）。
        filepath (str): 保存先のファイルパス。
    """
    # 方向ベクトルのX成分とY成分を保存
    save_main_out_image(vector[:, 0:1], filepath.replace('.png', '_x.png'))
    save_main_out_image(vector[:, 1:2], filepath.replace('.png', '_y.png'))
    
    # ベクトル場の大きさを可視化
    magnitude = torch.sqrt(vector[:, 0:1]**2 + vector[:, 1:2]**2)
    save_main_out_image(magnitude, filepath.replace('.png', '_magnitude.png'))
    
    # ベクトル場をカラーマッピングでHSV表現
    vec_np = vector.squeeze().cpu().numpy()  # [2, H, W]
    
    # 方向ベクトルの角度と大きさを計算
    direction = np.arctan2(vec_np[1], vec_np[0])  # [-pi, pi]の範囲
    magnitude_np = np.sqrt(vec_np[0]**2 + vec_np[1]**2)
    
    # HSVカラーマップ用の変換
    hue = ((direction + np.pi) / (2 * np.pi)) * 360  # 角度を[0, 360]の範囲に正規化
    saturation = np.ones_like(hue) * 0.9  # 彩度は固定
    value = 0.5 + 0.5 * np.minimum(magnitude_np / np.max(magnitude_np) if np.max(magnitude_np) > 0 else magnitude_np, 1.0)  # 明度は大きさに比例
    
    # HSV配列を作成
    hsv = np.stack([hue, saturation, value], axis=-1)
    
    # HSVからRGBに変換
    rgb = np.zeros((*hsv.shape[:-1], 3), dtype=np.float32)
    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
            h, s, v = hsv[i, j]
            rgb[i, j] = mcolors.hsv_to_rgb([h/360, s, v])
    
    # RGBを0-255に変換して保存
    rgb = (rgb * 255).astype(np.uint8)
    Image.fromarray(rgb).save(filepath)

def save_direction_vector_info(vector, mask, result_dir):
    """
    方向ベクトルの詳細情報を保存する関数。
    evaluate.pyと同様の方法で可視化。
    
    Parameters:
        vector (torch.Tensor): 方向ベクトル（形状: [B, 2, H, W]）。
        mask (torch.Tensor): マスク（形状: [B, 1, H, W]）。
        result_dir (str): 結果を保存するディレクトリパス。
    """
    # 通常の方向ベクトル可視化
    save_direction_vector_as_image(vector, os.path.join(result_dir, 'direction_vector.png'))
    
    # X成分とY成分を別々に保存
    save_main_out_image(vector[:, 0:1], os.path.join(result_dir, 'vec_x.png'))
    save_main_out_image(vector[:, 1:2], os.path.join(result_dir, 'vec_y.png'))
    
    # マスクされた方向ベクトル（血管部分のみの方向場）
    save_masked_output_with_imsave(vector[:, 0:1], mask, os.path.join(result_dir, 'masked_vec_x.png'))
    save_masked_output_with_imsave(vector[:, 1:2], mask, os.path.join(result_dir, 'masked_vec_y.png'))
    
    # 絶対値を取った方向ベクトル
    save_masked_output_with_imsave(torch.abs(vector[:, 0:1]), mask, os.path.join(result_dir, 'masked_abs_vec_x.png'), vmin=0, vmax=1)
    save_masked_output_with_imsave(torch.abs(vector[:, 1:2]), mask, os.path.join(result_dir, 'masked_abs_vec_y.png'), vmin=0, vmax=1)

def sample_single_image(args):
    """
    単一画像に対して異方性拡散をサンプリングし、各ステップを保存。
    
    Parameters:
        args: コマンドライン引数。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 出力ディレクトリの作成
    os.makedirs(args.result_dir, exist_ok=True)
    
    # データセットとモデルの準備
    transform_test = drive.get_transform(args, mode='test')
    testset = drive.DRIVEDataset("test", args.dataset_path, args.dataset_opt, transform=transform_test)
    
    # インデックスが範囲内か確認
    if args.image_index < 0 or args.image_index >= len(testset):
        print(f"指定されたインデックス {args.image_index} が範囲外です。0-{len(testset)-1} の範囲で指定してください。")
        return
    
    # モデルのロード
    model = Model(args).to(device)
    checkpoint = torch.load(args.pretrained_path, map_location=device)
    
    # キーがstate_dictかモデル自体かをチェック
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # DDPによる'module.'プレフィックスの処理
    # 'module.'プレフィックスがある場合、それを除去
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # 'module.'を除去
        new_state_dict[name] = v
    
    # 修正したstate_dictをロード
    model.load_state_dict(new_state_dict)
    
    model.eval()
    
    # 指定された単一画像の処理
    sample = testset[args.image_index]
    image = sample['transformed_image'].unsqueeze(0).to(device)  # バッチ次元を追加
    mask = sample['mask'].unsqueeze(0).to(device)
    
    # 画像と正解マスクを保存
    # drive_loader.pyを見ると、original_imageキーは存在しない
    # 代わりに、元の画像とマスクを保存
    input_image = sample['image']
    
    # テンソルの形状をチェックして適切に処理
    input_image_np = input_image.cpu().numpy()
    
    # 画像の次元を確認
    if len(input_image_np.shape) == 3:  # [C, H, W]
        if input_image_np.shape[0] == 1:  # グレースケール
            input_image_np = input_image_np.squeeze(0)  # [H, W]
        else:  # カラー画像
            input_image_np = input_image_np.transpose(1, 2, 0)  # [H, W, C]
    
    # 値を0-255に変換
    input_image_np = (input_image_np * 255).astype(np.uint8)
    Image.fromarray(input_image_np).save(os.path.join(args.result_dir, 'input_image.png'))
    
    save_tensor_as_image(mask, os.path.join(args.result_dir, 'ground_truth.png'))
    
    # 画像ファイル名の情報を保存（あれば）
    if 'meta' in sample and 'img_path' in sample['meta']:
        with open(os.path.join(args.result_dir, 'image_info.txt'), 'w') as f:
            f.write(f"Image path: {sample['meta']['img_path']}\n")
            f.write(f"Mask path: {sample['meta']['mask_path']}\n")
    
    with torch.no_grad():
        # モデルによる予測
        main_out, vec = model(image)
        vec = F.normalize(vec, p=2, dim=1)
        
        # 方向ベクトルの可視化
        save_direction_vector_info(vec, mask, args.result_dir)
        
        # 予測確率マップ
        original_size = mask.shape[2:]
        main_out_resized = F.interpolate(main_out, size=original_size, mode='bilinear', align_corners=False)
        preds = torch.sigmoid(main_out_resized)  # [B, 1, H, W]
        
        # 異方性拡散テンソルの生成
        diffusion_tensor = create_anisotropic_tensor_from_vector(vec)
        
        # 初期予測の保存
        save_tensor_as_image(preds, os.path.join(args.result_dir, 'initial_prediction.png'))
        save_tensor_as_image((preds > args.threshold).float(), os.path.join(args.result_dir, 'initial_mask.png'))
        
        # 異方性拡散を適用し、各ステップの結果を保存
        preds_anisotropic = anisotropic_diffusion(
            preds.clone(), 
            diffusion_tensor, 
            num_iterations=args.num_iterations, 
            gamma=args.gamma,
            save_dir=os.path.join(args.result_dir, 'diffusion_steps'),
            save_interval=args.save_interval
        )
        
        # 最終結果の保存
        save_tensor_as_image(preds_anisotropic, os.path.join(args.result_dir, 'final_prediction.png'))
        save_tensor_as_image((preds_anisotropic > args.threshold).float(), os.path.join(args.result_dir, 'final_mask.png'))
        
        # パディングを取り除いた結果を保存
        mask_unpadded = drive.unpad_to_original_by_size(mask)
        preds_unpadded = drive.unpad_to_original_by_size(preds)
        preds_anisotropic_unpadded = drive.unpad_to_original_by_size(preds_anisotropic)
        
        save_tensor_as_image(preds_unpadded, os.path.join(args.result_dir, 'initial_prediction_unpadded.png'))
        save_tensor_as_image((preds_unpadded > args.threshold).float(), os.path.join(args.result_dir, 'initial_mask_unpadded.png'))
        save_tensor_as_image(preds_anisotropic_unpadded, os.path.join(args.result_dir, 'final_prediction_unpadded.png'))
        save_tensor_as_image((preds_anisotropic_unpadded > args.threshold).float(), os.path.join(args.result_dir, 'final_mask_unpadded.png'))
        
        print(f"処理が完了しました。結果は {args.result_dir} に保存されました。")

def main():
    args = check_args()
    sample_single_image(args)

if __name__ == '__main__':
    main()
