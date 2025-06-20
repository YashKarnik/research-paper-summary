import os

from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    @staticmethod
    def get_file_upload_path(filename: str):
        base_dir = os.getenv("FILE_UPLOAD_PATH")
        print("base_dir=>")
        print(base_dir)
        return f"{base_dir}/{filename}"

    @staticmethod
    def get_index_upload_path(filename: str):
        base_dir = os.getenv("INDEX_UPLOAD_PATH")
        return f"{base_dir}/{filename}"
