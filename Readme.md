# Set-up

```bash
> virtualenv -p python3 venv
> source venv/bin/activate
(venv) > pip install -r requirements.txt
(venv) > jupyter notebook
```

# Sources

Information theoretic co-clustering based on
Mohammad Gorji Sefidmazgi
https://github.com/mgorjis/ITCC

Voting data is adapded from
[this Kaggle dataset](https://www.kaggle.com/datagraver/eurovision-song-contest-scores-19752017#eurovision_song_contest_1975_2017v4.xlsx),
amended with voting results from 1957-1974 and 2018 sourced from Wikipedia.

The shapefile used to create the map can be
[downloaded from Natural Earth Data](https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip).