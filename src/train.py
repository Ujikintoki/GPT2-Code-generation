import argparse
import math
import os

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 for Python Code Generation"
    )

    # 动态获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data_path = os.path.join(project_root, "data", "processed", "debug_sample")
    default_output_dir = os.path.join(project_root, "models", "checkpoints")

    # 数据与路径参数
    parser.add_argument(
        "--data_path",
        type=str,
        default=default_data_path,
        help="Path to the processed dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir,
        help="Directory to save model checkpoints",
    )

    # 消融实验核心参数：使用多大比例的数据进行训练 (1.0 = 100%, 0.5 = 50%, 0.1 = 10%)
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=1.0,
        help="Fraction of data to use for training (for scaling law ablation)",
    )

    # 训练超参数
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Training batch size per device"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    print("Starting training pipeline...")
    print(f"Ablation Setting: Using {args.data_fraction * 100}% of the dataset.")

    # 1. 加载数据与分词器
    print(f"Loading dataset from {args.data_path}...")
    dataset = load_from_disk(args.data_path)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 2. 实施消融实验逻辑 (切分 Data Fraction) & 划分验证集
    if args.data_fraction < 1.0:
        subset_size = int(len(dataset) * args.data_fraction)
        dataset = dataset.select(range(subset_size))
        print(f"Sliced dataset to {subset_size} samples for ablation study.")

    # 留出 10% 的数据作为验证集，用于在训练过程中计算真实的 Perplexity
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"Final Split -> Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # 3. 加载模型与数据整理器 (Data Collator)
    print("Loading GPT-2 model...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # DataCollator 负责将数据打包成 Batch。mlm=False 表示这是因果语言建模（预测下一个词），而非掩码语言建模（如 BERT）
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4. 配置训练参数 (兼容本地 Mac CPU/MPS 和云端 T4 GPU)
    # 自动检测是否在具备 CUDA (Colab GPU) 的环境中，如果是，开启 fp16 混合精度训练以节省一半显存
    fp16_enabled = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        # weight_decay=0.01,                 # 引入权重衰减防止过拟合
        # fp16=fp16_enabled,
        # eval_strategy="epoch",       # 每个 epoch 结束后在 eval_dataset 上评估一次
        # save_strategy="epoch",             # 每个 epoch 保存一次 Checkpoint
        # logging_steps=10,                  # 每 10 步打印一次日志
        load_best_model_at_end=False,  # 训练结束后自动加载验证集 Loss 最低的模型
        # metric_for_best_model="loss",
        # report_to="none"                   # 本地测试时关闭 W&B 避免弹窗提示
    )

    # 5. 初始化 Trainer 并启动训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("\nCommencing Fine-Tuning...")
    trainer.train()

    # 6. 计算最终的困惑度 (Perplexity) 并保存模型
    eval_results = trainer.evaluate()
    try:
        perplexity = math.exp(eval_results["eval_loss"])
    except OverflowError:
        perplexity = float("inf")

    print("\nTraining Complete!")
    print(f"Final Evaluation Loss: {eval_results['eval_loss']:.4f}")
    print(f"Final Perplexity (PPL): {perplexity:.4f}")

    # 将最终打磨好的模型权重保存到本地
    final_save_path = os.path.join(args.output_dir, "gpt2-python-final")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"Final model safely stored at: {final_save_path}")


if __name__ == "__main__":
    main()
