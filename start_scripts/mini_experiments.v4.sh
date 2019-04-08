# v3 - TriangleOpt with a max value equal to the max lr value from NoamOpt
signature=$1
# lr_freq=${5:-0} 
bert_dir='supply/rubert_cased_L-12_H-768_A-12_v3'
case "$signature" in

# mode_cls_sep
# mode_sep
# mode_without_cls_sep
# mode_without_cls_sep_bos

 "MINI_V4_TEST" )
 python train.py \
  --lr_warmup 16000 \
  --signature "$signature" \
  --segment_embedding 1 \
  --tie_weights 1 \
  --n_layers 3 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --tf_bert_model_load_from ${bert_dir}/bert_model.ckpt \
  --train_from "datasets/sber_srt_toloka_alice_samples.v3_tiny/*.valid.txt" \
  --valid_from "datasets/sber_srt_toloka_alice_samples.v3_tiny/*.valid.txt" \
  --batch_split 32 \
  --lm_weight 0.1 \
  --lr_freq 0 \
  --risk_weight 0 \
  --input_token_mode default \
  --spec_token_reinit '' \
  --n_epochs 5 \
  --max_seq_len 128 \
 ;;

 "MINI_V4_BASELINE_BT_L3_TW1_SEG1_V40_ShedLR0_SpecT0_SpecTReinit0_manSL128" )
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
  --input_token_mode default \
  --spec_token_reinit '' \
  --n_epochs 5 \
  --max_seq_len 128 \
 ;;

 "MINI_V4_BT_L3_TW1_SEG1_V40_ShedLR0_SpecT1_CS_SpecTReinit0_manSL128" )
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
  --spec_token_reinit '' \
  --n_epochs 5 \
  --max_seq_len 128 \
 ;;

 "MINI_V4_BT_L3_TW1_SEG1_V40_ShedLR0_SpecT1_S_SpecTReinit0_manSL128" )
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
  --spec_token_reinit '' \
  --n_epochs 5 \
  --max_seq_len 128 \
 ;;

 "MINI_V4_BT_L3_TW1_SEG1_V40_ShedLR0_SpecT1_deleteCS_SpecTReinit0_manSL128" )
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
  --input_token_mode SpecT1_deleteCS \
  --spec_token_reinit '' \
  --n_epochs 5 \
  --max_seq_len 128 \
 ;;

 "MINI_V4_BT_L3_TW1_SEG1_V40_ShedLR0_SpecT1_deleteCSB_SpecTReinit0_manSL128" )
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
  --input_token_mode SpecT1_deleteCSB \
  --spec_token_reinit '' \
  --n_epochs 5 \
  --max_seq_len 128 \
 ;;

 "MINI_V4_BT_L3_TW1_SEG1_V40_ShedLR0_SpecT1_CS_SpecTReinit1_EB_manSL128" )
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
  --n_epochs 5 \
  --max_seq_len 128 \
 ;;

 "MINI_V4_BT_L3_TW1_SEG1_V40_ShedLR0_SpecT1_CS_SpecTReinit1_B_manSL128" )
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
  --spec_token_reinit 'B' \
  --n_epochs 5 \
  --max_seq_len 128 \
 ;;

 "MINI_V4_BT_L3_TW1_SEG1_V40_ShedLR0_SpecTReinit1_B_manSL128" )
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
  --input_token_mode default \
  --spec_token_reinit 'B' \
  --n_epochs 5 \
  --max_seq_len 128 \
 ;;

 "MINI_V4_BT_L3_TW1_SEG1_V40_ShedLR0_SpecTReinit1_E_manSL128" )
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
  --input_token_mode default \
  --spec_token_reinit 'E' \
  --n_epochs 5 \
  --max_seq_len 128 \
 ;;

 "MINI_V4_BR_L3_TW1_SEG1_V40_ShedLR0_SpecTReinit0_manSL128" )
 python train.py \
  --lr_warmup 16000 \
  --signature "$signature" \
  --segment_embedding 1 \
  --tie_weights 1 \
  --n_layers 3 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --bare_model 1 \
  --train_from "datasets/sber_srt_toloka_alice_samples.v3_tiny/*.train.txt" \
  --valid_from "datasets/sber_srt_toloka_alice_samples.v3_tiny/*.valid.txt" \
  --batch_split 32 \
  --lm_weight 0.1 \
  --lr_freq 0 \
  --risk_weight 0 \
  --input_token_mode default \
  --spec_token_reinit '' \
  --n_epochs 5 \
  --max_seq_len 128 \
 ;;

 "MINI_V4_BT_L3_TW1_SEG0_V40_ShedLR0_SpecTReinit0_manSL128" )
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
  --input_token_mode default \
  --spec_token_reinit '' \
  --n_epochs 5 \
  --max_seq_len 128 \
 ;;

 "MINI_V4_BT_L3_TW1_SEG1_V40_ShedLR16_SpecTReinit0_manSL128" )
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
  --lr_freq 16 \
  --risk_weight 0 \
  --input_token_mode default \
  --spec_token_reinit '' \
  --n_epochs 5 \
  --max_seq_len 128 \
 ;;

 "MINI_V4_BT_L3_TW1_SEG1_V40_ShedLR64_SpecTReinit0_manSL128" )
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
  --lr_freq 64 \
  --risk_weight 0 \
  --input_token_mode default \
  --spec_token_reinit '' \
  --n_epochs 5 \
  --max_seq_len 128 \
 ;;

 "MINI_V4_BT_L3_TW1_SEG1_V40_ShedLR1_SpecTReinit0_manSL128" )
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
  --lr_freq 1 \
  --risk_weight 0 \
  --input_token_mode default \
  --spec_token_reinit '' \
  --n_epochs 5 \
  --max_seq_len 128 \
 ;;

 "MINI_V4_BT_L3_TW1_SEG1_V40_ShedLR0_SpecTReinit0_manSL256" )
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
  --batch_split 64 \
  --lm_weight 0.1 \
  --lr_freq 0 \
  --risk_weight 0 \
  --input_token_mode default \
  --spec_token_reinit '' \
  --n_epochs 5 \
  --max_seq_len 256 \
 ;;

esac
