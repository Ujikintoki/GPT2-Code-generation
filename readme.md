# GPT-2 python code generation project for CSIT5530, HKUST

## 1. Setup Environment

It is recommended to use a virtual environment to manage dependencies.

```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate the environment
# For macOS/Linux:
source venv/bin/activate
# For Windows:
venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt
```

# 2. train the model
```bash
# .py
python3 src/data_preprocess.py
# .ipynb
!python src/data_preprocess.py

#train
!python src/train.py \
    --data_path ./data/processed/full_dataset \
    --output_dir /content/drive/MyDrive/CSIT5530_Project/checkpoints/gpt2-full \
    # set 0.5 to train 50% data
    --data_fraction 1.0 \
    --epochs 3 \
    --batch_size 8 \
    --learning_rate 5e-5
```