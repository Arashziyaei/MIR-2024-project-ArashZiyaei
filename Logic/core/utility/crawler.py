import re
import time

import requests
from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json

# link to completed crawled_files.json: https://drive.google.com/file/d/1EzsMAh9AixZczzf23YBUGW2HA9WcwcVS/view?usp=sharing

class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15"
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        self.crawling_threshold = crawling_threshold
        self.not_crawled = []
        self.crawled = []
        self.added_ids = []
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        return URL.split('/')[4]

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        with open('crawled_files.json', 'w') as f:
            json.dump(list(self.crawled), f)

        with open('not_crawled_files.json', 'w') as f:
            json.dump(list(self.not_crawled), f)

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        with open('../crawled_files.json', 'r') as f:
            self.crawled = json.load(f)

        with open('../not_crawled_files.json', 'r') as f:
            self.not_crawled = json.load(f)

        for movie in self.crawled:
            id = movie['id']
            if id is not None and id not in self.added_ids:
                self.added_ids.append(id)
        for link in self.not_crawled:
            id = self.get_id_from_URL(link)
            if id is not None and id not in self.added_ids:
                self.added_ids.append(id)

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        response = requests.get(URL, headers=self.headers)
        while not response.ok:
            time.sleep(5)
            response = requests.get(URL, headers=self.headers)
        return response

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        # TODO update self.not_crawled and self.added_ids
        response = self.crawl(self.top_250_URL)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a'):
                str = link.get('href')
                if str.split('/')[1] == 'title' and str.split('/')[2] not in self.added_ids:
                    lst = ("https://www.imdb.com" + str).split('/')
                    del lst[5]
                    URL = '/'.join(lst)
                    URL = URL + '/'
                    id = self.get_id_from_URL(URL)
                    self.added_ids.append(id)
                    self.not_crawled.append(URL)
        else:
            print("response failed - extract_top_250")

    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
        TODO:
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """

        # help variables
        # WHILE_LOOP_CONSTRAINTS =
        # NEW_URL =
        # THERE_IS_NOTHING_TO_CRAWL = None

        self.extract_top_250()
        futures = []
        counter = 0

        with ThreadPoolExecutor(max_workers=20) as executor:
            while counter < self.crawling_threshold and len(self.not_crawled) > 0:
                with self.add_queue_lock:
                    URL = self.not_crawled[0]
                    del self.not_crawled[0]
                counter += 1
                # print('counter: ', counter)
                futures.append(executor.submit(self.crawl_page_info, URL))
                if not self.not_crawled:
                    wait(futures)
                    futures = []
            wait(futures)

    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.

        Parameters
        ----------
        URL: str
            The URL of the site
        """
        print("new iteration")
        response = self.crawl(URL)
        if response.status_code == 200:
            res = BeautifulSoup(response.content, 'html.parser')
            movie = self.get_imdb_instance()
            movie['id'] = self.get_id_from_URL(URL)
            self.extract_movie_info(res, movie, URL)
            with self.add_queue_lock:
                self.crawled.append(movie)
            for link in movie['related_links']:
                movie_id = self.get_id_from_URL(link)
                if movie_id not in self.added_ids and link not in self.not_crawled:
                    with self.add_queue_lock:
                        self.not_crawled.append(link)
                    with self.add_list_lock:
                        self.added_ids.append(movie_id)
        else:
            print('crawl_page_info failed: ', URL)

    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        movie['title'] = self.get_title(res)
        movie['first_page_summary'] = self.get_first_page_summary(res)
        movie['release_year'] = self.get_release_year(res)
        movie['mpaa'] = self.get_mpaa(res)
        movie['budget'] = self.get_budget(res)
        movie['gross_worldwide'] = self.get_gross_worldwide(res)
        movie['directors'] = self.get_director(res)
        movie['writers'] = self.get_writers(res)
        movie['stars'] = self.get_stars(res)
        movie['related_links'] = self.get_related_links(res)
        movie['genres'] = self.get_genres(res)
        movie['languages'] = self.get_languages(res)
        movie['countries_of_origin'] = self.get_countries_of_origin(res)
        movie['rating'] = self.get_rating(res)
        movie['summaries'] = self.get_summary(URL)
        movie['synopsis'] = self.get_synopsis(URL)
        movie['reviews'] = self.get_reviews_with_scores(URL)

    def get_summary_link(self, url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            return (url + 'plotsummary')
        except:
            print("failed to get summary link")

    def get_review_link(self, url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            return (url + 'reviews')
        except:
            print("failed to get review link")

    def get_title(self, soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            data = [x.text for x in soup.findAll("script", type="application/ld+json")]
            title = json.loads(data[0])['name']
            return title
        except:
            print("failed to get title")

    def get_first_page_summary(self, soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            first_page_summary = soup.find('span', {'class': "sc-466bb6c-2 chnFO"}).text
            return first_page_summary
        except:
            print("failed to get first page summary")

    def get_director(self, soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            directors = []
            data = soup.find('script', {'id': '__NEXT_DATA__', "type": "application/json"})
            for x in json.loads(data.contents[0])['props']['pageProps']['mainColumnData']['directors'][0]['credits']:
                directors.append(x['name']['nameText']['text'])
            return directors
        except:
            print("failed to get director")

    def get_stars(self, soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            data = [x.text for x in soup.findAll("script", type="application/ld+json")]
            stars = [x['name'] for x in json.loads(data[0])['actor']]
            return stars
        except:
            print("failed to get stars")

    def get_writers(self, soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            writers = []
            data = soup.find('script', {'id': '__NEXT_DATA__', "type": "application/json"})
            for x in json.loads(data.contents[0])['props']['pageProps']['mainColumnData']['writers'][0]['credits']:
                writers.append(x['name']['nameText']['text'])
            return writers
        except:
            print("failed to get writers")

    def get_related_links(self, soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            related_links = []
            data_tag = soup.findAll('a', {
                'class': 'ipc-poster-card__title ipc-poster-card__title--clamp-2 ipc-poster-card__title--clickable'})
            for data in data_tag:
                lst = ("https://www.imdb.com" + data['href']).split('/')
                del lst[5]
                URL = '/'.join(lst) + '/'
                # print("related links: ", URL)
                related_links.append(URL)
            return related_links
        except:
            print("failed to get related links")

    def get_summary(self, url):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            plotsummary_url = self.get_summary_link(url)
            response = self.crawl(plotsummary_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                data_tag = soup.find('script', {'id': '__NEXT_DATA__', "type": "application/json"})
                summaries = [ x ['htmlContent'] for x in
                              json.loads(data_tag.contents[0])['props']['pageProps']['contentData']['categories'][0][
                                  'section']['items']]
                return summaries
            else:
                print("response failed")
        except:
            print("failed to get summary")

    def get_synopsis(self, url):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            synopsis_url = self.get_summary_link(url)
            response = self.crawl(synopsis_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                data_tag = soup.find('script', {'id': '__NEXT_DATA__', "type": "application/json"})
                synopsis = [x['htmlContent'] for x in
                             json.loads(data_tag.contents[0])['props']['pageProps']['contentData']['categories'][1][
                                 'section']['items']]
                return synopsis
            else:
                print("response failed")
        except:
            print("failed to get synopsis")

    def get_reviews_with_scores(self, url):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            reviews_with_scores = []
            review_link = self.get_review_link(url)
            response = self.crawl(review_link)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                data_tag = soup.findAll('div', {'class': 'review-container'})
                for data in data_tag:
                    lst = []
                    for x in data.text.split('\n'):
                        if x != '' and x != ' ':
                            lst.append(x)

                    score = 'None'
                    review = None
                    pattern = r'^\d+(\.\d+)?/\d+$'

                    if re.match(pattern, lst[0]):
                        score = lst[0].split('/')[0]
                        del lst[1:3]
                    else:
                        del lst[1]

                    if lst[1][:8] == "Warning:":
                        review = lst[2]
                    else:
                        review = lst[1]

                    reviews_with_scores.append([review, score])
                return reviews_with_scores
            else:
                print("response failed")
        except:
            print("failed to get reviews")

    def get_genres(self, soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            data = [x.text for x in soup.findAll("script", type="application/ld+json")]
            genre = json.loads(data[0])['genre']
            return genre
        except:
            print("Failed to get generes")

    def get_rating(self, soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            data = [x.text for x in soup.findAll("script", type="application/ld+json")]
            rating = str(json.loads(data[0])['aggregateRating']['ratingValue'])
            return rating
        except:
            print("failed to get rating")

    def get_mpaa(self, soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            data = soup.find('script', {'id': '__NEXT_DATA__', "type": "application/json"})
            mpaa = json.loads(data.contents[0])['props']['pageProps']['aboveTheFoldData']['certificate']['rating']
            return mpaa
        except:
            print("failed to get mpaa")

    def get_release_year(self, soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            data = soup.find('script', {'id': '__NEXT_DATA__', "type": "application/json"})
            release_year = str(json.loads(data.contents[0])['props']['pageProps']['aboveTheFoldData']['releaseYear']['year'])
            return release_year
        except:
            print("failed to get release year")

    def get_languages(self, soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            data = soup.find('script', {'id': '__NEXT_DATA__', "type": "application/json"})
            languages = [x['text'] for x in
                         json.loads(data.contents[0])['props']['pageProps']['mainColumnData']['spokenLanguages']
                         ['spokenLanguages']]
            return languages
        except:
            print("failed to get languages")

    def get_countries_of_origin(self, soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            data = soup.find('script', {'id': '__NEXT_DATA__', "type": "application/json"})
            countriesOfOrigin = [x['text'] for x in
                         json.loads(data.contents[0])['props']['pageProps']['mainColumnData']['countriesOfOrigin'][
                             'countries']]
            return countriesOfOrigin
        except:
            print("failed to get countries of origin")

    def get_budget(self, soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            data = soup.find('script', {'id': '__NEXT_DATA__', "type": "application/json"})
            productionBudget = str(json.loads(data.contents[0])['props']['pageProps']['mainColumnData']['productionBudget']['budget']['amount'])
            return productionBudget
        except:
            print("failed to get budget")

    def get_gross_worldwide(self, soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            data = soup.find('script', {'id': '__NEXT_DATA__', "type": "application/json"})
            worldwideGross = str(json.loads(data.contents[0])['props']['pageProps']['mainColumnData']['worldwideGross']['total']['amount'])
            return worldwideGross
        except:
            print("failed to get gross worldwide")


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=1500)
    imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()


if __name__ == '__main__':
    main()
