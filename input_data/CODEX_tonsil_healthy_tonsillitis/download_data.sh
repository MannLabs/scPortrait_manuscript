# this script downloads the data for the following dataset:

# Demo dataset for: SPACEc, a streamlined, interactive Python workflow for multiplexed image processing and analysis

# https://datadryad.org/landing/show?id=doi%3A10.5061%2Fdryad.brv15dvj1

# If using this dataset please cite as:

# Tan, Yuqi; Kempchen, Tim (2024). Demo dataset for: SPACEc, a streamlined, interactive Python workflow for multiplexed image processing and analysis  [Dataset]. Dryad. https://doi.org/10.5061/dryad.brv15dvj1

# create subfolder for raw input data
mkdir -p raw_input_data

# download all required input files into raw_input_data
wget -O raw_input_data/read_me.md "https://datadryad.org/api/v2/files/3306734/download"
wget -O raw_input_data/example_data.zip "https://datadryad.org/api/v2/files/3306236/download"

# unzip input files inside raw_input_data
bsdtar -xf raw_input_data/example_data.zip -C raw_input_data && rm raw_input_data/example_data.zip