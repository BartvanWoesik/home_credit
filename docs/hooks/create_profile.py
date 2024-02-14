import pandas as pd
import os
from pathlib import Path
from ydata_profiling import ProfileReport

DOCS_PATH = Path("docs/assets")


def generate_markdown_script(name: str):
    python_script = f'''\
# Data Profile of Applprev_1_0

## Column description

{{{{ read_csv('assets/{name}_schema.csv') }}}}

## Data Statistics

<iframe width=2800, height=2000 frameBorder=0 src="../assets/{name}_report.html"></iframe>

    '''
    
    file_path = f"docs/{name}.md"
    if not os.path.exists(file_path):

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(python_script)
    else:
        print(f"The file '{name}.md' already exists in the 'docs' directory.")



base_wd = Path(os.getcwd())
data_path = base_wd / Path("data/parquet_files/train")
df_schema = pd.read_csv(base_wd / "data/feature_definitions.csv")
for file_name in os.listdir(data_path):
    base_name, _ = os.path.splitext(file_name)
    generate_markdown_script(base_name)

    if not os.path.exists(  DOCS_PATH / f"{base_name}_report.html"):
        df = pd.read_parquet(data_path /file_name)
        df_schema.set_index('Variable').T[df.columns.to_list()].T.to_csv(DOCS_PATH /  f'{base_name}_schema.csv')
        profile = ProfileReport(df, title=f"Profiling Report: {base_name}")
        profile.to_file(DOCS_PATH / f"{base_name}_report.html")