from pathlib import Path

import scipy
import pandas as pd
import scipy.sparse

def load_user_artists(user_artists_file: Path) -> scipy.sparse.csr_matrix:
    user_artists = pd.read_csv(user_artists_file, sep= "\t")
    user_artists.set_index(["userID","artistID"], inplace = True)
    coo = scipy.sparse.coo_matrix(
        
    )