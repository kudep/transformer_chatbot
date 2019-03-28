signature=$1
# lr_freq=${5:-0} 
bert_dir='supply/rubert_cased_L-12_H-768_A-12_v2'
case "$signature" in

 "bert_tw1_seg1_v40k_fin_0" )
 python train.py \
  --lr_warmup 16000 \
  --signature bert_tw1_seg1_v40k_fin_inf \
  --segment_embedding 1 \
  --tie_weights 1 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --tf_bert_model_load_from ${bert_dir}/model.ckpt-710142 \
  --train_from "datasets/toloka_alice_samples/*.train.txt" \
  --valid_from "datasets/toloka_alice_samples/*.valid.txt" \
  --batch_split 256 \
  --lm_weight 0.1 \
  --lr_freq 0 \
  --risk_weight 10 \
 ;;

 "bert_tw1_seg1_v40k_fin_8" )
 python train.py \
  --lr_warmup 16000 \
  --signature bert_tw1_seg1_v40k_fin_8 \
  --segment_embedding 1 \
  --tie_weights 1 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --tf_bert_model_load_from ${bert_dir}/model.ckpt-710142 \
  --train_from "datasets/toloka_alice_samples/*.train.txt" \
  --valid_from "datasets/toloka_alice_samples/*.valid.txt" \
  --batch_split 256 \
  --lm_weight 0.1 \
  --lr_freq 8 \
  --risk_weight 10 \
 ;;

 "bert_tw1_seg1_v40k_fin_4" )
 python train.py \
  --lr_warmup 16000 \
  --signature bert_tw1_seg1_v40k_fin_4 \
  --segment_embedding 1 \
  --tie_weights 1 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --tf_bert_model_load_from ${bert_dir}/model.ckpt-710142 \
  --train_from "datasets/toloka_alice_samples/*.train.txt" \
  --valid_from "datasets/toloka_alice_samples/*.valid.txt" \
  --batch_split 256 \
  --lm_weight 0.1 \
  --lr_freq 4 \
  --risk_weight 10 \
 ;;

 "bert_tw1_seg1_v40k_fin_2" )
 python train.py \
  --lr_warmup 16000 \
  --signature bert_tw1_seg1_v40k_fin_2 \
  --segment_embedding 1 \
  --tie_weights 1 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --tf_bert_model_load_from ${bert_dir}/model.ckpt-710142 \
  --train_from "datasets/toloka_alice_samples/*.train.txt" \
  --valid_from "datasets/toloka_alice_samples/*.valid.txt" \
  --batch_split 256 \
  --lm_weight 0.1 \
  --lr_freq 2 \
  --risk_weight 10 \
 ;;

 "bert_tw1_seg1_v40k_fin_1" )
 python train.py \
  --lr_warmup 16000 \
  --signature bert_tw1_seg1_v40k_fin_1 \
  --segment_embedding 1 \
  --tie_weights 1 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --tf_bert_model_load_from ${bert_dir}/model.ckpt-710142 \
  --train_from "datasets/toloka_alice_samples/*.train.txt" \
  --valid_from "datasets/toloka_alice_samples/*.valid.txt" \
  --batch_split 256 \
  --lm_weight 0.1 \
  --lr_freq 1 \
  --risk_weight 10 \
 ;;

 "bert_tw1_seg1_v40k_fin_0.5" )
 python train.py \
  --lr_warmup 16000 \
  --signature bert_tw1_seg1_v40k_fin_0.5 \
  --segment_embedding 1 \
  --tie_weights 1 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --tf_bert_model_load_from ${bert_dir}/model.ckpt-710142 \
  --train_from "datasets/toloka_alice_samples/*.train.txt" \
  --valid_from "datasets/toloka_alice_samples/*.valid.txt" \
  --batch_split 256 \
  --lm_weight 0.1 \
  --lr_freq 0.5 \
  --risk_weight 10 \
 ;;

 "bert_tw1_seg1_v40k_fin_0.25" )
 python train.py \
  --lr_warmup 16000 \
  --signature bert_tw1_seg1_v40k_fin_0.25 \
  --segment_embedding 1 \
  --tie_weights 1 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --tf_bert_model_load_from ${bert_dir}/model.ckpt-710142 \
  --train_from "datasets/toloka_alice_samples/*.train.txt" \
  --valid_from "datasets/toloka_alice_samples/*.valid.txt" \
  --batch_split 256 \
  --lm_weight 0.1 \
  --lr_freq 0.25 \
  --risk_weight 10 \
 ;;

 "bert_tw1_seg1_v40k_fin_0.125" )
 python train.py \
  --lr_warmup 16000 \
  --signature bert_tw1_seg1_v40k_fin_0.125 \
  --segment_embedding 1 \
  --tie_weights 1 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --tf_bert_model_load_from ${bert_dir}/model.ckpt-710142 \
  --train_from "datasets/toloka_alice_samples/*.train.txt" \
  --valid_from "datasets/toloka_alice_samples/*.valid.txt" \
  --batch_split 256 \
  --lm_weight 0.1 \
  --lr_freq 0.125 \
  --risk_weight 10 \
 ;;

 "bare_tw1_seg1_v40k_fin_0" )
 python train.py \
  --lr_warmup 16000 \
  --signature bare_tw1_seg1_v40k \
  --segment_embedding 1 \
  --tie_weights 1 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --bare_model 1 \
  --train_from "datasets/toloka_alice_samples/*.train.txt" \
  --valid_from "datasets/toloka_alice_samples/*.valid.txt" \
  --batch_split 128 \
  --lm_weight 0.1 \
  --lr_freq 0 \
  --risk_weight 0 \
 ;;
esac


# bert_tw1_seg1_v40k_fin_inf
# bert_tw1_seg1_v40k_fin_8
# bert_tw1_seg1_v40k_fin_4
# bert_tw1_seg1_v40k_fin_2
# bert_tw1_seg1_v40k_fin_1
# bert_tw1_seg1_v40k_fin_0.5
# bert_tw1_seg1_v40k_fin_0.25
# bert_tw1_seg1_v40k_fin_0.125
