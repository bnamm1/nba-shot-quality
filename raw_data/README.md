# raw_data/

Raw NBA Stats API v3 play-by-play files are not stored in this repository. They come from [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data).

Download all twelve seasons (2013–14 through 2024–25) into this folder:

```bash
for y in $(seq 2013 2024); do
  curl -L -o raw_data/nbastatsv3_$y.tar.xz \
    https://github.com/shufinskiy/nba_data/raw/main/datasets/nbastatsv3_$y.tar.xz
  tar -xJf raw_data/nbastatsv3_$y.tar.xz -C raw_data && rm raw_data/nbastatsv3_$y.tar.xz
done
```

Each archive extracts to `nbastatsv3_YYYY.csv`, where `YYYY` is the season's starting year.

These files are needed by `enrich_shots.py`. Section *8)* of the Model B notebook also reads `nbastatsv3_2024.csv` to compute actual win totals. The enriched shot data the models use is already in `enriched_data/` (Git LFS).
