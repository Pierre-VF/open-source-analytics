"""
This is a script that helps generating the metadata of a number of organisations

You will need to do the following in order for this to work:

1) Install the requirements in `requirements.txt` in your local python environment (works for python>=3.12)
2) Copy the `.env-example` file in a `.env` file and add your Mistral AI API key

Notes:

- Caching is used on disk to avoid re-iterating expensive calls
- Markdown prompts are used to prompt the LLM, this makes it easier to read and finetune them without being an expert

"""

import json
import os
import os.path
import warnings
from typing import Any
from urllib.request import urlretrieve

import diskcache
import pandas as pd
import pydantic_settings
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from pydantic_ai import Agent, ModelHTTPError
from pydantic_ai.models.mistral import MistralModel
from tqdm import tqdm

# ------------------------------------------------------------------------------------
# Common configuration
# ------------------------------------------------------------------------------------


def render_from_template(template: str, context: dict[str, Any]) -> str:
    x = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "prompts"))
    ).get_template(template)
    return x.render(context)


class Settings(pydantic_settings.BaseSettings):
    MISTRAL_API_KEY: str
    MISTRAL_MODEL: str = "mistral-medium"
    DISK_CACHE_DIRECTORY: str = ".data"
    INPUT_FOLDER: str = ".data/inputs"
    OUTPUT_FOLDER: str = ".data/outputs"

    def ensure_folders_exist(self) -> None:
        os.makedirs(self.INPUT_FOLDER, exist_ok=True)
        os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(self.DISK_CACHE_DIRECTORY, exist_ok=True)

    def get_mistral_api_key(self) -> str:
        if self.MISTRAL_API_KEY is None:
            raise EnvironmentError("No MISTRAL_API_KEY defined in environment")
        return self.MISTRAL_API_KEY

    @property
    def disk_cache(self) -> diskcache.Cache:
        if self.DISK_CACHE_DIRECTORY is None:
            raise EnvironmentError("No DISK_CACHE_DIRECTORY defined in environment")
        return diskcache.Cache(directory=os.path.expanduser(self.DISK_CACHE_DIRECTORY))


def _f_download_if_missing(url: str, target: str) -> None:
    if not os.path.exists(target):
        urlretrieve(url, target)


organisations_xlsx_url = r"https://api.getgrist.com/o/docs/api/docs/gSscJkc5Rb1Rw45gh1o1Yc/download/xlsx?viewSection=7&tableId=Organizations&activeSortSpec=%5B-156%5D&filters=%5B%7B%22colRef%22%3A124%2C%22filter%22%3A%22%7B%5C%22excluded%5C%22%3A%5B%5D%7D%22%7D%5D&linkingFilter=%7B%22filters%22%3A%7B%7D%2C%22operations%22%3A%7B%7D%7D"


# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

# Configuring the environment
load_dotenv()
settings = Settings()
settings.ensure_folders_exist()
cache = settings.disk_cache
mm = MistralModel(settings.MISTRAL_MODEL)
llm_agent = Agent(mm)

input_file = f"{settings.INPUT_FOLDER}/orgs.xlsx"
output_file = f"{settings.OUTPUT_FOLDER}/orgs_classified.json"
output_file_csv = output_file.replace(".json", ".csv")

# ------------------------------------------------------------------------------------

# Downloading the organisation file if not found locally
_f_download_if_missing(organisations_xlsx_url, input_file)

# ------------------------------------------------------------------------------------

df_orgs = pd.read_excel(input_file)
print(" ")
print(f"Found {len(df_orgs)} organisations to process")
print(" ")


def _f(url) -> dict:
    c = cache.get(url)
    if c and (c.get("exception") is None):
        return c
    try:
        prompt = render_from_template(
            "organisation_metadata_generation_simple.md",
            {"WEBSITE": url},
        )
        x = llm_agent.run_sync(prompt)
        clean_x = x.output.replace("\n", "").replace("```", "").replace("json{", "{")
        out = json.loads(clean_x)
    except ModelHTTPError as e:
        if e.status_code in [401, 403]:
            warnings.warn(f"LLM model error (due to user permissions {e})")
        out = dict(exception=str(e))
    except Exception as e:
        out = dict(exception=str(e))
    out["url"] = url
    cache.add(url, out)
    return out


# ------------------------------------------------------------------------------------

# Iterating the enhancement over the website URLs
x_out = []
for i, r in tqdm(df_orgs.iterrows()):
    x = r["organization_website"]
    if isinstance(x, str) and x.startswith("https://"):
        x_out.append(_f(x))


# ------------------------------------------------------------------------------------

# Writing the results in JSON and CSV
with open(output_file, "w") as f:
    json.dump(x_out, f)

with open(output_file, "r") as f:
    x_dict = json.load(f)

df_out = pd.DataFrame(x_dict).merge(
    df_orgs[
        [
            "organization_website",
            "form_of_organization",
            "location_country",
        ]
    ].rename(
        columns={
            "organization_website": "url",
            "form_of_organization": "manual_Type",
            "location_country": "manual_Location",
        }
    ),
    how="left",
    on="url",
)
df_out[
    ["url", "Confidence", "manual_Type", "Type", "manual_Location", "Location"]
].sort_values("Confidence", ascending=False).to_csv(output_file_csv)

# ------------------------------------------------------------------------------------
print(f"Metadata generation completed (see output in {output_file_csv})")
# ------------------------------------------------------------------------------------

#
# Notes: here is a quick snapshot of the issues witnessed in LLm outputs
#
#   - https://digitalearthafrica.org': {  "Location": {    "Country": "GLOBAL",    "Continent": "AF"  // Primary operational focus on Africa, but global partnerships  },  "Type": "Non-profit",  "Confidence": 0.98}
#
#
