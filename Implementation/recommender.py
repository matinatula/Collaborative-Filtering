# Recommendation using the implicit library

from pathlib import Path
from typing import Tuple, List

import implicit
import implicit.recommender_base
import scipy

from data import load_user_artists, ArtistRetriever


# computing recommendations for a given user using the implicit library with two attributes (artist_retriever and implicit_model)
class ImplicitRecommender:
    def __init__(
        self,
        artist_retriever: ArtistRetriever,
        implicit_model: implicit.recommender_base.RecommenderBase,
    ):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model

    # fitting model to the user-artists matrix
    def fit(self, user_artists_matrix: scipy.sparse.csr_matrix) -> None:
        self.implicit_model.fit(user_artists_matrix)

    # return top n recommendations for the given user
    def recommend(
            self, user_id: int,
            user_artists_matrix: scipy.sparse.csr_matrix,
            n: int = 10,
    ) -> Tuple[List[str], List[float]]:
        artist_ids, scores = self.implicit_model.recommend(
            user_id, user_artists_matrix[user_id], N=n
        )
        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]
        return artists, scores


if __name__ == "__main__":

    # load user-artists matrix
    user_artists = load_user_artists(
        Path("../Dataset/user_artists.dat"))

    # instantiate artist retriever
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(
        Path("../Dataset/artists.dat"))

    # instantiate ALS using implicit
    implicit_model = implicit.als.AlternatingLeastSquares(
        factors=50, iterations=10, regularization=0.01
    )

    # instantiate recommender,fit, and recommend
    recommender = ImplicitRecommender(artist_retriever, implicit_model)
    recommender.fit(user_artists)
    artists, scores = recommender.recommend(2, user_artists, n=5)

    # print results
    for artist, score in zip(artists, scores):
        print(f"{artist}: {score}")
