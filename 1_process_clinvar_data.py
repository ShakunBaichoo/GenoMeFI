import os
import gzip
import xml.etree.ElementTree as ET
import csv
import subprocess
import pandas as pd

class ClinVarProcessingPipeline:
    def __init__(self, base_dir="./data", fasta_path="./Ref38Genome/GCF_000001405.40_GRCh38.p14_genomic.fna"):
        self.base_dir = base_dir
        self.fasta_path = fasta_path

    def extract_from_xml(self, input_xml, output_csv, fields):
        """
        Step 1: Create trimmed version of dataset from XML.
        """
        with gzip.open(input_xml, "rb") as fin, open(output_csv, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=fields.keys())
            writer.writeheader()
            context = ET.iterparse(fin, events=("end",))
            for event, elem in context:
                if elem.tag != "VariationArchive":
                    continue
                row = {}
                for col, (path, attr) in fields.items():
                    if path.startswith("@"):
                        row[col] = elem.get(path[1:], "")
                    else:
                        node = elem.find(path)
                        if node is None:
                            row[col] = ""
                        elif attr is None or attr == "text":
                            row[col] = (node.text or "").strip()
                        else:
                            row[col] = node.get(attr, "")
                writer.writerow(row)
                elem.clear()

    def extract_clinsig_from_vcf(self, vcf_path, output_tsv):
        """
        Step 2: Parse VCF file to extract Clinical Significance information.
        """
        cmd = (
            f"module load samtools; "
            f"(echo -e 'CHROM\\tPOS\\tREF\\tALT\\tCLNSIG'; "
            f"bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\t%INFO/CLNSIG\\n' {vcf_path}) "
            f"> {output_tsv}"
        )
        subprocess.run(cmd, shell=True, check=True)
        print(f"Saved CLNSIG info to {output_tsv}")

    def merge_trimmed_and_clinsig(self, trimmed_csv, clinsig_tsv, merged_csv):
        """
        Step 3: Merge Clinical significance with Trimmed ClinVar data.
        """
        trimmed = pd.read_csv(trimmed_csv, dtype=str)
        clinsig = pd.read_csv(clinsig_tsv, sep="\t", dtype=str)
        clinsig = clinsig.rename(columns={
            "CHROM": "AssemblyChr",
            "POS":   "Start",
            "REF":   "RefAlleleVCF",
            "ALT":   "AltAlleleVCF"
        })
        merged = pd.merge(
            trimmed, clinsig,
            on=["AssemblyChr", "Start", "RefAlleleVCF", "AltAlleleVCF"],
            how="left"
        )
        merged.to_csv(merged_csv, index=False)
        print(merged_csv,"shape:",merged.shape)
        print(f"Merged CSV saved to {merged_csv}")

    def categorize_clinsig(self, merged_csv, out_csv):
        """
        Step 4: Categorize Clinical Significance Labels and create binary labels.
        """
        df = pd.read_csv(merged_csv, dtype=str)

        def map_group(cln):
            if pd.isna(cln) or cln in ('.', 'not_provided', 'no_classifications_from_unflagged_records'):
                return "no_assertion"
            v = cln.lower()
            if "conflicting_classifications_of_pathogenicity" in v:
                return "conflicting"
            if "uncertain_significance" in v:
                return "uncertain_significance"
            if "pathogenic" in v and "benign" not in v:
                return "pathogenic"
            if "benign" in v and "pathogenic" not in v:
                return "benign"
            if v.startswith("drug_response") or "|drug_response" in v:
                return "drug_response"
            if v.startswith("risk_factor") or "|risk_factor" in v:
                return "risk_factor"
            if v.startswith("protective") or "confers_sensitivity" in v:
                return "protective"
            if v.startswith("association") or "association" in v:
                return "association"
            if "risk_allele" in v:
                return "risk_allele"
            return "other"

        df["is_pathogenic"] = df["CLNSIG"].str.contains("pathogenic", case=False, na=False)
        df["clinvar_group"] = df["CLNSIG"].apply(map_group)
        df.to_csv(out_csv, index=False)
        print(out_csv,"shape:",df.shape)
        print(f"Categorized ClinSig and saved to {out_csv}")

    def drop_ambiguous_clinsig_variants(self, input_csv, output_csv, group_col="clinvar_group"):
        """
        Step 5: Drop 'other' and 'uncertain_significance' records **early** for all further steps.
        """
        df = pd.read_csv(input_csv, dtype=str)
        filtered = df[~df[group_col].isin(["other","conflicting","uncertain_significance","no_assertion"])].copy()
        print(f"Dropped {(len(df) - len(filtered))} variants of type 'other' or 'uncertain_significance' or 'no assertion'.")
        filtered.to_csv(output_csv, index=False)
        print(filtered,"shape:",filtered.shape)
        print(f"Saved filtered data to {output_csv}")

    def create_windows_and_fasta(self, filtered_csv, window_sizes):
        """
        Step 6: For each window size, create BED and FASTA in its folder.
        """
        df = pd.read_csv(filtered_csv, dtype=str).dropna(subset=["CanonicalSPDI", "Start", "Stop"])
        for WINDOW in window_sizes:
            folder = f"{self.base_dir}/windows_{WINDOW}"
            os.makedirs(folder, exist_ok=True)
            bedfile = f"{folder}/windows_{WINDOW}.bed"
            fastafile = f"{folder}/clinvar_windows_{WINDOW}.fa"

            with open(bedfile, "w") as out:
                for _, row in df.iterrows():
                    contig = row["CanonicalSPDI"].split(":", 1)[0]
                    S = int(row["Start"])
                    E = int(row["Stop"])
                    bed_start = max(0, S - WINDOW - 1)
                    bed_end = E + WINDOW
                    var_id = row["VariationID"]
                    out.write(f"{contig}\t{bed_start}\t{bed_end}\tvar{var_id}\n")
            print(f"Wrote BED file: {bedfile}")

            cmd = [
                "bedtools", "getfasta",
                "-fi", self.fasta_path,
                "-bed", bedfile,
                "-fo", fastafile,
                "-name+"
            ]
            print(f"Extracting FASTA: {fastafile}")
            subprocess.run(cmd, check=True)
            print(f"Saved: {fastafile}\n")

    def integrate_fasta_to_table(self, filtered_csv, window_sizes):
        """
        Step 7: For each window, merge FASTA to filtered variant table and save per-window annotated file.
        """
        for WINDOW in window_sizes:
            folder = f"{self.base_dir}/windows_{WINDOW}"
            fastafile = f"{folder}/clinvar_windows_{WINDOW}.fa"
            out_csv = f"{folder}/clinvar_windows_{WINDOW}_annotated.csv"
            annotated = pd.read_csv(filtered_csv, dtype=str)
            records = []
            with open(fastafile) as fh:
                lines = [l.strip() for l in fh if l.strip()]
            for i in range(0, len(lines), 2):
                header = lines[i][1:]
                seq = lines[i+1]
                if "::" in header:
                    var_part, coord_part = header.split("::", 1)
                elif "|" in header:
                    var_part, coord_part = header.split("|", 1)
                else:
                    raise ValueError(f"Unrecognized header format: {header}")
                variant_id = var_part.replace("var", "")
                contig, span = coord_part.split(":", 1)
                start, end = map(int, span.split("-", 1))
                records.append({
                    "VariationID": variant_id,
                    "Contig": contig,
                    "WindowStart": start,
                    "WindowEnd": end,
                    "Sequence": seq
                })
            fasta_df = pd.DataFrame(records)
            merged = pd.merge(annotated, fasta_df, on="VariationID", how="inner")
            merged.to_csv(out_csv, index=False)
            print(f"Merged annotated CSV with sequences to {out_csv}")

    def clean_sequences(self, in_csv, out_csv):
        """
        Step 8: Convert all sequences to upper-case and drop those not ACGT only.
        """
        df = pd.read_csv(in_csv, dtype=str)
        df["Sequence"] = df["Sequence"].str.upper()
        bad_mask = ~df["Sequence"].str.fullmatch(r"[ACGT]+")
        df_final = df[~bad_mask].reset_index(drop=True)
        df_final.to_csv(out_csv, sep="\t", index=False)
        print(f"Cleaned data written to {out_csv}")

    """
    Step 9: Create a subset of variants for Pathogenic category.
    Generates a balanced binary train/test split (30,000 train, 3,000 test).
    """
    def create_binary_subsets(self, cleaned_csv, out_train, out_test,
                                     n_path_train=15000, n_nonpath_train=15000,
                                     n_path_test=1500, n_nonpath_test=1500):
        df = pd.read_csv(cleaned_csv, sep="\t")
        df['is_pathogenic'] = (df['clinvar_group'] == 'pathogenic').astype(int)
        df['clinvar_binary'] = df['clinvar_group'].apply(
            lambda x: 'pathogenic' if x == 'pathogenic' else 'non-pathogenic'
        )
    
        # Shuffle within each class
        df_path = df[df['clinvar_binary'] == 'pathogenic'].sample(frac=1, random_state=42)
        df_nonpath = df[df['clinvar_binary'] == 'non-pathogenic'].sample(frac=1, random_state=42)
    
        # Train and test splits
        train_path = df_path.iloc[:n_path_train]
        test_path = df_path.iloc[n_path_train : n_path_train + n_path_test]
        train_nonpath = df_nonpath.iloc[:n_nonpath_train]
        test_nonpath = df_nonpath.iloc[n_nonpath_train : n_nonpath_train + n_nonpath_test]
    
        # Concatenate and shuffle
        train_binary = pd.concat([train_path, train_nonpath]).sample(frac=1, random_state=42).reset_index(drop=True)
        test_binary = pd.concat([test_path, test_nonpath]).sample(frac=1, random_state=42).reset_index(drop=True)
    
        # Save to file
        train_binary.to_csv(out_train, sep='\t', index=False)
        test_binary.to_csv(out_test, sep='\t', index=False)
    
        print("Train class counts:\n", train_binary['clinvar_binary'].value_counts())
        print("Test class counts:\n", test_binary['clinvar_binary'].value_counts())

pipeline = ClinVarProcessingPipeline(
    base_dir="./data",
    fasta_path="./Ref38Genome/GCF_000001405.40_GRCh38.p14_genomic.fna"
)
FIELDS = {
    "VariationID":        ("@VariationID", None),
    "VariationName":      ("./ClassifiedRecord/SimpleAllele/Name", "text"),
    "VariantType":        ("./ClassifiedRecord/SimpleAllele/VariantType", "text"),
    "CanonicalSPDI":      ("./ClassifiedRecord/SimpleAllele/CanonicalSPDI", "text"),
    "GeneSymbol":         ("./ClassifiedRecord/SimpleAllele/GeneList/Gene", "Symbol"),
    "OMIM":               ("./ClassifiedRecord/SimpleAllele/GeneList/Gene/OMIM", "text"),
    "AssemblyChr":        ("./ClassifiedRecord/SimpleAllele/Location/SequenceLocation[@Assembly='GRCh38']", "Chr"),
    "Start":              ("./ClassifiedRecord/SimpleAllele/Location/SequenceLocation[@Assembly='GRCh38']", "start"),
    "Stop":               ("./ClassifiedRecord/SimpleAllele/Location/SequenceLocation[@Assembly='GRCh38']", "stop"),
    "RefAlleleVCF":       ("./ClassifiedRecord/SimpleAllele/Location/SequenceLocation[@Assembly='GRCh38']", "referenceAlleleVCF"),
    "AltAlleleVCF":       ("./ClassifiedRecord/SimpleAllele/Location/SequenceLocation[@Assembly='GRCh38']", "alternateAlleleVCF"),
    "VariantLength":      ("./ClassifiedRecord/SimpleAllele/Location/SequenceLocation[@Assembly='GRCh38']", "variantLength"),
    "ProteinChange":      ("./ClassifiedRecord/SimpleAllele/ProteinChange", "text"),
    "HGVS_coding":        ("./ClassifiedRecord/SimpleAllele/HGVSlist/HGVS[@Type='coding']/NucleotideExpression/Expression", "text"),
    "HGVS_protein":       ("./ClassifiedRecord/SimpleAllele/HGVSlist/HGVS[@Type='coding']/ProteinExpression/Expression", "text"),
    "DateLastUpdated":    ("@DateLastUpdated", None),
    "NumberOfSubmitters": ("@NumberOfSubmitters", None),
    "RecordType":         ("@RecordType", None)
}

WINDOW_SIZES = [200, 300]

# File names
input_xml = "./clinvar_xml_data/ClinVarVCVRelease_2025-06.xml.gz"
trimmed_csv = "./data/clinvar_trimmed.csv"
vcf_path = "./clinvar_vcf/clinvar.vcf.gz"
clinsig_tsv = "./data/clinvar_clinsig.tsv"
annotated_csv = "./data/clinvar_annotated.csv"
catSig_csv = "./data/clinvar_annotated_catSig.csv"
filtered_csv = "./data/completeClinvarFinal.csv"

# 1. Extract basic variant information from XML
pipeline.extract_from_xml(input_xml, trimmed_csv, FIELDS)

# 2. Extract clinical significance from VCF
pipeline.extract_clinsig_from_vcf(vcf_path, clinsig_tsv)

# 3. Merge basic info and clinical significance
pipeline.merge_trimmed_and_clinsig(trimmed_csv, clinsig_tsv, annotated_csv)

# 4. Categorize clinical significance and add labels
pipeline.categorize_clinsig(annotated_csv, catSig_csv)

# 5. Drop those missing any assertion, conflicting, ambiguous/other/uncertain significance variants
pipeline.drop_ambiguous_clinsig_variants(catSig_csv, filtered_csv)

# 6. For each window size, create window BED/FASTA files
pipeline.create_windows_and_fasta(filtered_csv, window_sizes=WINDOW_SIZES)

# 7. For each window, merge FASTA to variant table
pipeline.integrate_fasta_to_table(filtered_csv, window_sizes=WINDOW_SIZES)

# 8. For each window, clean sequences and create binary train/test splits
for WINDOW in WINDOW_SIZES:
    folder = f"./data/windows_{WINDOW}"
    annotated_file = f"{folder}/clinvar_windows_{WINDOW}_annotated.csv"
    cleaned_file = f"{folder}/clinvar_windows_{WINDOW}_cleaned.tsv"
    train_file = f"{folder}/clinvar_binary_train_{WINDOW}.tsv"
    test_file = f"{folder}/clinvar_binary_test_{WINDOW}.tsv"
    pipeline.clean_sequences(annotated_file, cleaned_file)
    pipeline.create_binary_subsets(cleaned_file, train_file, test_file) 