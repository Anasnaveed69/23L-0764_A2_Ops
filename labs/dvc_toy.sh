#!/usr/bin/env bash
# Task 1.4 toy part — run step by step (copy each block) in Git Bash / a Linux terminal,
# taking the screenshots marked [SHOT]. Needs: git, dvc (pip install "dvc[s3]").
# Before starting create an EMPTY DagsHub repo named 23L-0764_A2_Ops-dvc-toy and set:
#   export DAGSHUB_USER=<your dagshub username>
#   export DAGSHUB_TOKEN=<your dagshub token>
set -euo pipefail

# ---- 1. new empty folder, Git + DVC ------------------------------------------------------
mkdir -p ~/dvc-toy && cd ~/dvc-toy
git init
dvc init                         # creates .dvc/ (config + cache dir) and .dvcignore
git commit -m "init DVC"

# ---- 2. small CSV, tracked by DVC ----------------------------------------------------------
printf "id,score\n1,0.91\n2,0.42\n3,0.77\n" > scores.csv
dvc add scores.csv               # moves content to .dvc/cache, writes scores.csv.dvc pointer
cat scores.csv.dvc               # the pointer: md5 + size + path   [SHOT]
git add scores.csv.dvc .gitignore
git commit -m "data v1: 3 rows"

# ---- 3. change it, version 2 ---------------------------------------------------------------
printf "4,0.13\n5,0.66\n" >> scores.csv
dvc add scores.csv
git add scores.csv.dvc
git commit -m "data v2: 5 rows"
git log --oneline                #                                     [SHOT]

# ---- 4. go back to the older version ------------------------------------------------------
git checkout HEAD~1 -- scores.csv.dvc   # old pointer
dvc checkout                            # DVC swaps the file for the cached v1 content
cat scores.csv                          # 3 rows again                [SHOT]
git checkout HEAD -- scores.csv.dvc && dvc checkout   # back to v2

# ---- 5. remote storage on DagsHub, push, delete, pull -------------------------------------
git remote add origin "https://dagshub.com/$DAGSHUB_USER/23L-0764_A2_Ops-dvc-toy.git"
dvc remote add -d origin "s3://dvc"
dvc remote modify origin endpointurl "https://dagshub.com/$DAGSHUB_USER/23L-0764_A2_Ops-dvc-toy.s3"
dvc remote modify origin --local access_key_id "$DAGSHUB_TOKEN"     # --local = .dvc/config.local (git-ignored)
dvc remote modify origin --local secret_access_key "$DAGSHUB_TOKEN"
git add .dvc/config && git commit -m "configure DagsHub DVC remote"
git push -u origin HEAD:main     # (username = DagsHub user, password = token)
dvc push --all-commits           # upload the data of EVERY committed version (v1 and v2)
rm scores.csv && rm -rf .dvc/cache                       # simulate a fresh machine   [SHOT]
ls; dvc status -c || true
dvc pull                         # downloads from DagsHub and restores scores.csv
cat scores.csv                   #                                                    [SHOT]
