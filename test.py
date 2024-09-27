import yaml
from box.config_box import ConfigBox
from ccs.entity.config_entity import DataIngestionConfig

path_to_yaml = "/home/thaihocit02/thaihoc/code/project-iast/end-to-end-mlops/config/config.yaml"

with open(path_to_yaml) as yaml_file:
    content = yaml.safe_load(yaml_file)
    config = ConfigBox(content)

thaihoc = config.data_ingestion

data_ingestion_config = DataIngestionConfig(
            root_dir=thaihoc.root_dir,
            data_URL=thaihoc.data_URL,
            local_file_data=thaihoc.local_data_file,
            unzip_dir=thaihoc.unzip_dir
        )

print(data_ingestion_config)