import feedparser
import pandas as pd
import os


class DataCollector:
    """
    A class that collects news titles related to a specific subject and the user response/sentiment to said titles.
    """

    def __init__(self, output_file: str) -> None:
        """
        Constructs all necessary attributes for a data_collector object.

        Parameter(s)
        ------------
        output_file: str
            The name of the csv file to output the labelled data to.
        
        Returns
        -------
        None
        """

        self.subject_txt = input("Enter subject: ").lower()
        self.subject_url = self.subject_txt.split(" ")
        self.subject_url = "%20".join(self.subject_url)
        self.news_feed_url = f"https://news.google.com/rss/search?q={self.subject_url}&hl=en-US&gl=US&ceid=US:en"
        self.news = feedparser.parse(self.news_feed_url)
        self.entries = self.news.entries
        self.n = len(self.entries)
        self.news_data_file = os.path.join("Data", output_file)
        self.old_titles = []
        self.new_observations = []
        
    def check_old_titles(self) -> None:
        """
        Recollects all previously seen news titles.

        Parameter(s)
        ------------
        None
        
        Returns
        -------
        None
        """
        
        try:
            news_df = pd.read_csv(self.news_data_file)
            self.old_titles = list(news_df["title"])
        except FileNotFoundError as not_found:
            print(f"{not_found.filename} doesn't exist yet")
    
    def get_user_input(self) -> None:
        """
        Prompts the user for a subject, then shows the user one news title at a time related to said subject.
        The user is expected to respond with their sentiment.

        Parameter(s)
        ------------
        None
        
        Returns
        -------
        None
        """
        
        print(f"{self.n} titles found")
        print("provide responses for the news article titles as they appear")
        print("enter 1 for legitimate news, 0 for all others (i.e. clickbait), s to skip, and q to quit")
        for entry in self.entries:
            if entry.title not in self.old_titles:
                response = input(f"{entry.title}: ")
                if response in ("1","0"):
                    self.new_observations.append([self.subject_txt, entry.title, response])
                    self.old_titles.append(entry.title)
                elif response == "s":
                    continue
                else:
                    break

    def output_to_file(self) -> None:
        """
        Stores the new news titles and user responses in a csv file.
        If the csv file does not yet exist, a new one is created.

        Parameter(s)
        ------------
        None
        
        Returns
        -------
        None
        """
        
        new_observations_df = pd.DataFrame(self.new_observations, columns=["topic", "title", "response"])
        new_observations_df.to_csv(self.news_data_file, mode="a",
                                   header=(not os.path.exists(self.news_data_file)),
                                   index=False)