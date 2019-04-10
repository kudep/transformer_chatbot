# v3 - TriangleOpt with a max value equal to the max lr value from NoamOpt
signature=$1
# lr_freq=${5:-0} 
bert_dir='supply/rubert_cased_L-12_H-768_A-12_v3'
case "$signature" in

# mode_cls_sep
# mode_sep
# mode_without_cls_sep
# mode_without_cls_sep_bos

 "MINI_V5_BT_L3_TW1_SEG1_V40_ShedLR0_SpecT1_CS_SpecTReinit1_EB_manSL128" )
 python train.py \
  --lr_warmup 16000 \
  --signature "$signature" \
  --segment_embedding 1 \
  --tie_weights 1 \
  --n_layers 3 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --tf_bert_model_load_from ${bert_dir}/bert_model.ckpt \
  --train_from "datasets/sber_srt_toloka_alice_samples.v3_tiny/*.train.txt" \
  --valid_from "datasets/sber_srt_toloka_alice_samples.v3_tiny/*.valid.txt" \
  --batch_split 32 \
  --lm_weight 0.1 \
  --lr_freq 0 \
  --risk_weight 0 \
  --input_token_mode SpecT1_CS \
  --spec_token_reinit 'EB' \
  --n_epochs 20 \
  --max_seq_len 128 \
 ;;

 "MINI_V5_BT_L3_TW1_SEG0_V40_ShedLR0_SpecT1_CS_SpecTReinit1_EB_manSL128" )
 python train.py \
  --lr_warmup 16000 \
  --signature "$signature" \
  --segment_embedding 0 \
  --tie_weights 1 \
  --n_layers 3 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --tf_bert_model_load_from ${bert_dir}/bert_model.ckpt \
  --train_from "datasets/sber_srt_toloka_alice_samples.v3_tiny/*.train.txt" \
  --valid_from "datasets/sber_srt_toloka_alice_samples.v3_tiny/*.valid.txt" \
  --batch_split 32 \
  --lm_weight 0.1 \
  --lr_freq 0 \
  --risk_weight 0 \
  --input_token_mode SpecT1_CS \
  --spec_token_reinit 'EB' \
  --n_epochs 20 \
  --max_seq_len 128 \
 ;;

 "MINI_V5_BT_L3_TW1_SEG1_V40_ShedLR0_SpecT1_S_SpecTReinit1_EB_manSL128" )
 python train.py \
  --lr_warmup 16000 \
  --signature "$signature" \
  --segment_embedding 1 \
  --tie_weights 1 \
  --n_layers 3 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --tf_bert_model_load_from ${bert_dir}/bert_model.ckpt \
  --train_from "datasets/sber_srt_toloka_alice_samples.v3_tiny/*.train.txt" \
  --valid_from "datasets/sber_srt_toloka_alice_samples.v3_tiny/*.valid.txt" \
  --batch_split 32 \
  --lm_weight 0.1 \
  --lr_freq 0 \
  --risk_weight 0 \
  --input_token_mode SpecT1_S \
  --spec_token_reinit 'EB' \
  --n_epochs 20 \
  --max_seq_len 128 \
 ;;

 "MINI_V5_BT_L3_TW1_SEG0_V40_ShedLR0_SpecT1_S_SpecTReinit1_EB_manSL128" )
 python train.py \
  --lr_warmup 16000 \
  --signature "$signature" \
  --segment_embedding 0 \
  --tie_weights 1 \
  --n_layers 3 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --tf_bert_model_load_from ${bert_dir}/bert_model.ckpt \
  --train_from "datasets/sber_srt_toloka_alice_samples.v3_tiny/*.train.txt" \
  --valid_from "datasets/sber_srt_toloka_alice_samples.v3_tiny/*.valid.txt" \
  --batch_split 32 \
  --lm_weight 0.1 \
  --lr_freq 0 \
  --risk_weight 0 \
  --input_token_mode SpecT1_S \
  --spec_token_reinit 'EB' \
  --n_epochs 20 \
  --max_seq_len 128 \
 ;;

esac
