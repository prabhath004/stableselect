# Kaggle P100 Rerun Workflow

Use this workflow to recreate the pilot result files. The script writes zip
checkpoints to `/kaggle/working` after each benchmark pair.

## 1. Start Notebook

Enable GPU and confirm:

```python
!nvidia-smi
```

## 2. Clone Repo

```python
!git clone https://github.com/prabhath004/stableselect.git
%cd stableselect
!git log --oneline -1
```

## 3. Install P100 Stack

```python
!pip uninstall -y torch torchvision torchaudio triton bitsandbytes
!pip install --no-cache-dir -r requirements-kaggle-p100.txt
```

## 4. Hugging Face Login

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

hf_token = UserSecretsClient().get_secret("HF_TOKEN")
login(token=hf_token)
```

## 5. Run Pilot

```python
!bash scripts/run_kaggle_p100_pilot.sh
```

The script creates these files:

```text
/kaggle/working/stableselect_after_english.zip
/kaggle/working/stableselect_after_hindi.zip
/kaggle/working/stableselect_after_arabic.zip
/kaggle/working/stableselect_after_spanish.zip
/kaggle/working/stableselect_final_pilot.zip
```

Download each zip as it appears. At minimum, download:

```text
/kaggle/working/stableselect_final_pilot.zip
```

## 6. Download Link

```python
from IPython.display import FileLink
FileLink("/kaggle/working/stableselect_final_pilot.zip")
```

