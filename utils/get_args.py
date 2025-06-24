import argparse
import datetime
import os

def get_args():
    parser = argparse.ArgumentParser(description="Arguments for PromptHash Training and Evaluation")

    # --- Hashing Parameters ---
    parser.add_argument("--k-bits-list", type=str, default="16,32,64", help="Comma-separated list of hash code lengths.")
    parser.add_argument("--auxiliary-bit-dim", type=int, default=256, help="Length of auxiliary hash codes.")

    # --- Model Architecture ---
    parser.add_argument("--top-k-label", type=int, default=8, help="Top-k operation in LTA module (if applicable).")
    parser.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu"], help="Activation function.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")
    parser.add_argument("--res-mlp-layers", type=int, default=2, help="Number of ResMLP blocks.")
    parser.add_argument("--concept-num", type=int, default=196, help="Number of concepts (e.g., patch embeddings).")
    parser.add_argument("--transformer-layers", type=int, default=1, help="Number of transformer layers in specific modules.")
    parser.add_argument("--is-freeze-clip", type=lambda x: (str(x).lower() == 'true'), default=True, help="Freeze CLIP backbone (True/False).")

    # --- Training & Validation Parameters ---
    parser.add_argument("--is-train", action="store_true", help="Set this flag to run training.")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per GPU.")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--valid-freq", type=int, default=1, help="Validate every N epochs.")
    parser.add_argument("--num-workers", type=int, default=20, help="Number of dataloader workers.")
    parser.add_argument("--seed", type=int, default=1814, help="Random seed for reproducibility.")

    # --- Optimizer Parameters ---
    parser.add_argument("--clip-lr", type=float, default=1e-6, help="Learning rate for CLIP parameters.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Base learning rate for other modules.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for optimizer.")
    parser.add_argument("--warmup-proportion", type=float, default=0.05, help="Proportion of training steps for LR warmup.")

    # --- Mixed Precision (AMP) ---
    parser.add_argument("--use-amp", default="false", help="Enable Automatic Mixed Precision (AMP).")
    parser.add_argument("--amp-dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"], help="Mixed precision dtype (bf16 or fp16).")

    # --- Loss Weights & Hyperparameters ---
    parser.add_argument("--tao-global", type=float, default=0.07, help="Temperature for global prompt alignment loss.")
    parser.add_argument("--tao-local", type=float, default=0.07, help="Temperature for local prompt alignment loss.")
    parser.add_argument("--hyper-recon", type=float, default=0.005, help="Weight for reconstruction loss.")
    parser.add_argument("--hyper-global", type=float, default=5.0, help="Weight for global prompt alignment loss.")
    parser.add_argument("--hyper-local", type=float, default=5.0, help="Weight for local prompt alignment loss.")
    parser.add_argument("--hyper-lambda", type=float, default=1.0, help="Overall weight for prompt alignment losses.")
    parser.add_argument("--hyper-alpha2", type=float, default=0.7, help=".")
    parser.add_argument("--hyper-margin", type=float, default=0.2, help=".")
    parser.add_argument("--hyper-shift", type=float, default=0.1, help=".")
    parser.add_argument("--mu", type=float, default=20.0, help="Weight multiplier for auxiliary bit losses.")
    parser.add_argument("--hyper-cls-inter", type=float, default=5.0, help="Weight for inter-class affinity loss.")
    parser.add_argument("--hyper-cls-intra", type=float, default=0.005, help="Weight for intra-class affinity loss.")
    parser.add_argument("--hyper-quan", type=float, default=1.0, help="Weight for quantization loss.")

    # --- Dataset & File Paths ---
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "flickr25k", "nuswide"], help="Dataset name.")
    parser.add_argument("--query-num", type=int, default=5000, help="Number of query samples.")
    parser.add_argument("--train-num", type=int, default=10000, help="Number of training samples.")
    parser.add_argument("--pretrained", type=str, default="", help="Path to pretrained model weights.")
    parser.add_argument("--index-file", type=str, default="index.mat", help="Name of the index file within dataset folder.")
    parser.add_argument("--caption-file", type=str, default="caption.mat", help="Name of the caption file within dataset folder.")
    parser.add_argument("--label-file", type=str, default="label.mat", help="Name of the label file within dataset folder.")
    parser.add_argument("--max-words", type=int, default=77, help="Maximum number of words in text input.")
    parser.add_argument("--resolution", type=int, default=224, help="Image input resolution.")
    parser.add_argument("--result-name", type=str, default="results", help="Base directory name for saving results.")

    # --- System / Misc ---
    parser.add_argument("--rank", type=int, default=0, help="GPU rank (for distributed training, default 0 for single GPU).")

    # --- Parse Arguments ---
    args = parser.parse_args()

    # --- Post-processing Arguments ---
    # Create save directory path
    _time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_type = "train" if args.is_train else "test"
    k_bits_str = args.k_bits_list.replace(",", "-") # e.g., "16-32-64"

    # Add AMP info to path if used
    amp_str = f"_amp-{args.amp_dtype}" if args.use_amp else ""

    save_dir_name = f"{args.dataset}_{k_bits_str}bits_{run_type}{amp_str}_{_time}"
    args.save_dir = os.path.join(".", args.result_name, save_dir_name)

    return args


if __name__ == '__main__':
    # Example of how to use it
    args = get_args()
    print("Parsed Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"\nSave directory: {args.save_dir}")

    # Example check for training mode
    if args.is_train:
        print("\nRunning in Training Mode")
        if args.use_amp:
            print(f"Using Automatic Mixed Precision with dtype: {args.amp_dtype}")
    else:
        print("\nRunning in Evaluation/Test Mode")
        if args.pretrained:
            print(f"Loading pretrained model from: {args.pretrained}")
        else:
            print("Warning: Running in test mode without a --pretrained model specified.")

