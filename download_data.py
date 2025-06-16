import os
import requests
from tqdm import tqdm
import hashlib

def download_file(url, dest_folder="."):
    """Download a file with a progress bar."""
    os.makedirs(dest_folder, exist_ok=True)
    local_filename = os.path.join(dest_folder, url.split("/")[-1])
    if os.path.exists(local_filename):
        print(f"File already exists: {local_filename}")
        return local_filename
    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(local_filename, 'wb') as file, tqdm(
        desc=local_filename, total=total, unit='B', unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=1024*1024):
            size = file.write(data)
            bar.update(size)
    return local_filename

def md5sum(filename):
    """Compute the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def check_md5(filename, md5_file):
    """Check if the MD5 sum matches what's in the .md5 or checksum file."""
    if not os.path.exists(md5_file):
        print(f"No MD5 file found: {md5_file}")
        return False
    # Find the line for our filename (supporting both .md5 and md5checksums.txt formats)
    md5_expected = None
    with open(md5_file, "r") as f:
        for line in f:
            if filename.split("/")[-1] in line:
                md5_expected = line.split()[0]
                break
    if md5_expected is None:
        print(f"MD5 not found for {filename}")
        return False
    md5_actual = md5sum(filename)
    if md5_actual == md5_expected:
        print(f"MD5 OK for {filename}")
        return True
    else:
        print(f"MD5 mismatch for {filename}!\n  Expected: {md5_expected}\n  Actual:   {md5_actual}")
        return False

# ==============================
# 1. Download ClinVar Data
# ==============================
clinvar_xml_urls = [
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/ClinVarVCVRelease_2025-06.xml.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/ClinVarVCVRelease_2025-06.xml.gz.md5"
]
clinvar_vcf_urls = [
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz",
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.md5",
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi"
]

# ==============================
# 2. Download Human Reference Genome
# ==============================
ref_urls = [
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz",
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/md5checksums.txt"
]

download_dir = "./data/"
# Download all files
for url in clinvar_xml_urls + clinvar_vcf_urls + ref_urls:
    download_file(url, dest_folder=download_dir)

# ==============================
# 3. MD5 Checking
# ==============================
# Check MD5 for XML
xml_file = os.path.join(download_dir, "clinvar_xml_data/ClinVarVCVRelease_2025-06.xml.gz")
xml_md5 = os.path.join(download_dir, "clinvar_xml_data/ClinVarVCVRelease_2025-06.xml.gz.md5")
check_md5(xml_file, xml_md5)

# Check MD5 for VCF
vcf_file = os.path.join(download_dir, "clinvar_vcf/clinvar.vcf.gz")
vcf_md5 = os.path.join(download_dir, "clinvar_vcf/clinvar.vcf.gz.md5")
check_md5(vcf_file, vcf_md5)

# Check MD5 for Reference FASTA (uses large md5checksums.txt file)
fasta_file = os.path.join(download_dir, "Ref38Genome/GCF_000001405.40_GRCh38.p14_genomic.fna.gz")
fasta_md5 = os.path.join(download_dir, "Ref38Genome/md5checksums.txt")
check_md5(fasta_file, fasta_md5)

print("All downloads and checks complete.")