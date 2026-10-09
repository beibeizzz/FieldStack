#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
NGPUS="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"

TRAIN_ARGS=(
    --concat_senet_dims "1024,256"
    --pair_pool mean
    --concat_mlp_dims "1024,512,256,128"
    --rankmixer_tokenize semantic
    --rankmixer_group_tokens "5,3,1,1,1,1"
    --rankmixer_seq_per_token 1
    --rankmixer_dim 1152
    --rankmixer_heads 0
    --rankmixer_layers 2
    --rankmixer_expansion 2
    --rankmixer_pool mean
    --rankmixer_norm rmsnorm
    --dense_per_fid
    --user_dense_groups "61|87|89,90,91,118|120|121|123|130|131|132"
    --item_dense_groups "124|127,128|129"
    --loss_type bce_pos_weight
    --pos_weight 2.0
    --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:1024"
    --seq_dual_window_domains "seq_d"
    --seq_stca_domains "seq_d"
    --stca_layers 2
    --stca_heads 4
    --stca_expansion 2
    --two_stream_mlp 1
    --two_stream_dims "512,128"
    --two_stream_input senet
    --two_stream_fusion bilinear
    --bilinear_groups 8
    --optimizer muon
    --muon_lr 5e-4
    --muon_momentum 0.95
    --muon_weight_decay 0.01
    --muon_ns_steps 5
    --use_swa
    --swa_start_epoch 2
    --swa_collect_every 200
    --swa_bn_batches 50
    --use_sam
    --sam_rho 0.05
    --emb_skip_threshold 1000000
    --num_workers 8
    --d_model 64
    --emb_dim 64
    --hidden_mult 4
    --dropout_rate 0.01
    --batch_size 1024
    --no_amp
    --longtail_transform log1p
    --weighted_pair_pool_fids "62,63,64,65,66"
    --longtail_fids "62,63,64,65,66"
    --save_every_epoch
)

echo "FieldStack: ${NGPUS} visible GPU(s)"
if [[ "${NGPUS}" -gt 1 ]]; then
    exec "${PYTHON_BIN}" -m torch.distributed.run --standalone \
        --nproc_per_node="${NGPUS}" "${CODE_DIR}/train.py" "${TRAIN_ARGS[@]}" "$@"
else
    exec "${PYTHON_BIN}" -u "${CODE_DIR}/train.py" "${TRAIN_ARGS[@]}" "$@"
fi
