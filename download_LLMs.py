import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

### Note that I have saved my models in a folder one level up as I will be using it
### for other projects, apart from this technical assessment.
### So you may decide to save it in a more appropriate place.
### But then specify that path when you are running the finteuning.

# Example for Nucleotide Transformer (NT)
# Ensure base folder exists
base_dir = "./hugging_face_models"
os.makedirs(base_dir, exist_ok=True)

# Nucleotide Transformer
nt_model = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
nt_save_dir = os.path.join(base_dir, "nucleotide-transformer-500m-human-ref")
os.makedirs(nt_save_dir, exist_ok=True)
tokenizer_nt = AutoTokenizer.from_pretrained(nt_model)
model_nt = AutoModelForSequenceClassification.from_pretrained(nt_model)
tokenizer_nt.save_pretrained(nt_save_dir)
model_nt.save_pretrained(nt_save_dir)

# DNABERT6
dnabert6_model = "zhihan1996/DNA_bert_6"
dnabert6_save_dir = os.path.join(base_dir, "DNABERT-6")
os.makedirs(dnabert6_save_dir, exist_ok=True)
tokenizer_dnabert6 = AutoTokenizer.from_pretrained(dnabert6_model, trust_remote_code=True)
model_dnabert6 = AutoModel.from_pretrained(dnabert6_model, trust_remote_code=True)
tokenizer_dnabert6.save_pretrained(dnabert6_save_dir)
model_dnabert6.save_pretrained(dnabert6_save_dir)

# GROVER
grover_model = "PoetschLab/GROVER"
grover_save_dir = os.path.join(base_dir, "GROVER_genomic")
os.makedirs(grover_save_dir, exist_ok=True)
tokenizer_grover = AutoTokenizer.from_pretrained(grover_model)
model_grover = AutoModelForSequenceClassification.from_pretrained(grover_model)
tokenizer_grover.save_pretrained(grover_save_dir)
model_grover.save_pretrained(grover_save_dir)