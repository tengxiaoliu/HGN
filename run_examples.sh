#!/usr/bin/env bash

python scripts/5_dump_features_v1.py --para_path 'data/dataset/data_processed/dev_distractor/multihop_para.json' \
      --raw_data 'data/dataset/data_raw/hotpot_dev_distractor_v1.json' \
      --model_name_or_path roberta-large \
      --do_lower_case --ner_path 'data/dataset/data_processed/dev_distractor/ner.json' \
      --model_type roberta \
      --tokenizer_name roberta-large \
      --output_dir 'data/dataset/data_graph' \
      --doc_link_ner 'data/dataset/data_processed/dev_distractor/doc_link_ner.json'