from huggingface_hub import snapshot_download
import os
# 在代码最开头设置国内镜像源，解决网络问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def download_all_models():
    # 获取当前脚本所在目录的父目录作为项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(project_root, "pretrained_weights")

    print(f"开始通过国内镜像源下载模型到: {base_dir}...")

    # 1. 下载 VAE 模型
    print("\n[1/3] 正在下载 sd-vae-ft-mse...")
    snapshot_download(
        repo_id="stabilityai/sd-vae-ft-mse",
        local_dir=os.path.join(base_dir, "sd-vae-ft-mse"),
        local_dir_use_symlinks=False,
        resume_download=True
    )

    # 2. 下载 SD 变体模型 (EchoMimic 依赖)
    print("\n[2/3] 正在下载 sd-image-variations-diffusers...")
    snapshot_download(
        repo_id="lambdalabs/sd-image-variations-diffusers",
        local_dir=os.path.join(base_dir, "sd-image-variations-diffusers"),
        local_dir_use_symlinks=False,
        resume_download=True
    )

    # 3. 下载 EchoMimic 核心权重
    print("\n[3/3] 正在下载 EchoMimic 核心权重...")
    snapshot_download(
        repo_id="BadToBest/EchoMimic",
        local_dir=base_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )

    print("\n✅ 所有模型下载完毕！请检查 pretrained_weights 文件夹。")


if __name__ == "__main__":
    download_all_models()
