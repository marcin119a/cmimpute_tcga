#https://www.kaggle.com/datasets/martininf1n1ty/dna-methylation-imputed
import kagglehub

# wget https://www.kaggle.com/api/v1/datasets/download/martininf1n1ty/dna-methylation-imputed?dataset_version_number=3
# Try with force_download to get the latest version
# Or specify a different version if you know it exists: "martininf1n1ty/dna-methylation-imputed/versions/1"
path = kagglehub.dataset_download("martininf1n1ty/dna-methylation-imputed", force_download=True)

print(f"Dataset downloaded to: {path}")