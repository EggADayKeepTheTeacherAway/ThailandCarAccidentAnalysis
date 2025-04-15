import gdown

URL = "https://drive.google.com/drive/folders/1dR8YyJNAkb7YELQ_EY55hpT7XrNL2-7X?usp=sharing"
OUTPUT_FOLDER = "data"


class DataGetter:
    def __init__(self, url: str, dest_folder: str = "data") -> None:
        self.url = url
        self.dest_folder = dest_folder

    def get_data(self) -> None:
        """
        This function is used to get the data from the given URL.
        :return: None
        """
        print(f"Fetching data from {self.url} to {self.dest_folder}")

        gdown.download_folder(
            self.url, 
            output=self.dest_folder, 
            quiet=False,
            )

        print("Data fetched successfully")


if __name__ == "__main__":
    data_getter = DataGetter(URL, OUTPUT_FOLDER)
    data_getter.get_data()
