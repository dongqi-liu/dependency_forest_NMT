#!/bin/bash
#Set job requirements
#SBATCH -p gpu_titanrtx_shared
#SBATCH --gpus-per-node=1
#SBATCH -t 120:00:00 

#Loading modules
module load 2020
module load Python/3.8.2-GCCcore-9.3.0
# pip install --user -U spacy
# python -m spacy download en_core_web_sm --user
# python -m spacy download de_core_news_sm --user
# pip install --user sacrebleu
# pip install --user torchtext
# pip install --user --upgrade git+git://github.com/nltk/nltk.git
# pip install --user numpy
# pip install --user tensorboard
#create output directory
mkdir "$TMPDIR"/output_dir

#Run program
python iwslt_forest_en_zh.py "$TMPDIR"/output_dir

#Copy output data from scratch to home
cp -r "$TMPDIR"/output_dir $HOME