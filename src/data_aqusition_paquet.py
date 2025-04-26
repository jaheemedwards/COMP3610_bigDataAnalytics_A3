import duckdb
import pyarrow as pa
import tarfile
import tempfile
import os
from datasets import Dataset
import glob

# === CONFIG ===
base_path = r"D:\Jaheem\jaheemProjetToDelete\COMP3610_bigDataAnalytics_A3-master\src\data\raw"
join_key = "parent_asin"

# Set a custom temporary directory to avoid C: drive space issues
tempfile.tempdir = r"D:\Jaheem\jaheemProjetToDelete\COMP3610_bigDataAnalytics_A3-master\src\data\temp"

# === Auto-detect categories from review tar.gz files ===
review_tars = glob.glob(os.path.join(base_path, "raw_review_*.tar.gz"))
categories = []

for review_tar in review_tars:
    cat = os.path.basename(review_tar).replace("raw_review_", "").replace(".tar.gz", "")
    meta_tar = os.path.join(base_path, f"raw_meta_{cat}.tar.gz")
    if os.path.exists(meta_tar):
        categories.append(cat)

if not categories:
    raise RuntimeError("❌ No matching category tar.gz files found!")

print(f"🎉 Detected categories: {categories}\n")

# === Create DuckDB tables ===
def create_tables():
    duckdb.sql("DROP TABLE IF EXISTS review_stream")
    duckdb.sql("DROP TABLE IF EXISTS meta_stream")

    print("🛠️ Creating DuckDB tables...")

    duckdb.sql("""
        CREATE TABLE review_stream (
            rating DOUBLE,
            title TEXT,
            text TEXT,
            images TEXT,
            asin TEXT,
            parent_asin TEXT,
            user_id TEXT,
            timestamp BIGINT,
            helpful_vote BIGINT,
            verified_purchase BOOLEAN
        );
    """)

    duckdb.sql("""
        CREATE TABLE meta_stream (
            main_category TEXT,
            title TEXT,
            average_rating DOUBLE,
            rating_number BIGINT,
            features TEXT,
            description TEXT,
            price TEXT,
            images TEXT,
            videos TEXT,
            store TEXT,
            categories TEXT,
            details TEXT,
            parent_asin TEXT,
            bought_together TEXT,
            subtitle TEXT,
            author TEXT
        );
    """)

# === Extract and stream Arrow files from tar.gz into DuckDB ===
def stream_from_tar_to_duckdb(table_name, tar_path):
    print(f"🔄 Processing tar file: {tar_path}")

    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.name.endswith(".arrow"):
                print(f"  📦 Extracting: {member.name}")
                file_obj = tar.extractfile(member)

                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(file_obj.read())
                    print(f"    📝 Processed file: {tmp_file.name}")

                    dataset = Dataset.from_file(tmp_file.name)
                    reader = dataset.data.to_reader()
                    for batch in reader:
                        table = pa.Table.from_batches([batch])
                        duckdb.register("tmp_batch", table)
                        duckdb.sql(f"INSERT INTO {table_name} SELECT * FROM tmp_batch")
                        duckdb.unregister("tmp_batch")

                try:
                    os.remove(tmp_file.name)
                    print(f"    ✅ Deleted temporary file: {tmp_file.name}")
                except PermissionError as e:
                    print(f"    ⚠️ Could not delete temporary file: {e}")

# === Process each category ===
for category in categories:
    print(f"\n🔄 Processing category: {category}")

    review_tar = os.path.join(base_path, f"raw_review_{category}.tar.gz")
    meta_tar   = os.path.join(base_path, f"raw_meta_{category}.tar.gz")
    output_path = os.path.join(base_path, f"joined_{category}.parquet")

    create_tables()
    print(f"📚 Ingesting review data...")
    stream_from_tar_to_duckdb("review_stream", review_tar)

    print(f"📦 Ingesting metadata...")
    stream_from_tar_to_duckdb("meta_stream", meta_tar)

    print(f"🔗 Joining review and meta on '{join_key}' and exporting...")
    duckdb.sql(f"""
        COPY (
            SELECT '{category}' AS category, r.*, m.*
            FROM review_stream r
            JOIN meta_stream m ON r.{join_key} = m.{join_key}
        ) TO '{output_path}' (FORMAT PARQUET);
    """)

    print(f"✅ Exported: {output_path}\n")
