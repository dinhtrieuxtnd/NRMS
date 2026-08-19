# NRMS

Pipeline PyTorch triển khai Neural News Recommendation with Multi-Head
Self-Attention (NRMS) trên MINDsmall. Project hỗ trợ preprocessing, training,
resume checkpoint, test evaluation và recommendation theo lịch sử đọc.

Mô hình hiện dùng title của news và GloVe 840B 300 chiều. Entity/relation
embedding đi kèm MIND không được sử dụng.

## Cài đặt

Từ project root:

```powershell
python -m pip install -r requirement.txt
```

Tải 3 file [MINDsmall_train.zip](https://huggingface.co/datasets/yjw1029/MIND/tree/main), [MINDsmall_dev.zip](https://huggingface.co/datasets/yjw1029/MIND/tree/main) và [glove.840B.300d.txt](https://www.kaggle.com/datasets/takuok/glove840b300dtxt?resource=download)

Chuẩn bị dữ liệu theo cấu trúc:

```text
data/raw/
  MINDsmall_train/
    behaviors.tsv
    news.tsv
  MINDsmall_dev/
    behaviors.tsv
    news.tsv
  glove/
    glove.840B.300d.txt
```

Pipeline không tự tải MINDsmall hoặc GloVe.

## Chạy pipeline

### 1. Preprocess

```powershell
python -m scripts.preprocess --config configs/preprocess.yaml
python -m scripts.validate_processed `
  --data-dir data/processed/mindsmall_nrms_v1
```

Thêm `--overwrite` vào lệnh preprocess khi cần tạo lại processed dataset.

### 2. Train

```powershell
python -m scripts.train --config configs/train.yaml
```

Mỗi run được ghi vào:

```text
outputs/<experiment_name>/YYYY-MM-DD_HH-MM-SS/
```

### 3. Evaluate test set

```powershell
python -m scripts.evaluate `
  --run-dir outputs/mindsmall_nrms_v1/YYYY-MM-DD_HH-MM-SS
```

Kết quả gồm test metrics và prediction cho từng impression/news.

### 4. Recommend

```powershell
python -m scripts.recommend `
  --run-dir outputs/mindsmall_nrms_v1/YYYY-MM-DD_HH-MM-SS `
  --history N123 N456 N789 `
  --top-k 10
```

Giới hạn catalog bằng `--candidates N100 N200 N300`. Thêm
`--output recommendations.json` để ghi kết quả ra file. CLI tự loại các news
đã xuất hiện trong history.

## Smoke test

[configs/train_smoke.yaml](configs/train_smoke.yaml) là cấu hình nhẹ để kiểm tra toàn bộ
đường train/validation trên processed dataset hiện có:

```powershell
python -m scripts.train `
  --config configs/train_smoke.yaml `
  --max-train-batches 1 `
  --max-validation-batches 1
```

Hai giới hạn batch là tham số CLI. Nếu bỏ chúng, config vẫn chạy toàn bộ một
epoch.

Processed dataset hiện dùng embedding 300 chiều, vì vậy
`model.embedding_dim` phải bằng `300`, kể cả khi dùng model nhỏ để smoke test.

## Resume training

Tăng `training.epochs` trong config tới tổng số epoch mong muốn, sau đó chạy:

```powershell
python -m scripts.train `
  --config configs/train.yaml `
  --resume outputs/mindsmall_nrms_v1/YYYY-MM-DD_HH-MM-SS/checkpoints/last.pt
```

Resume tiếp tục trong run directory cũ và khôi phục model, optimizer, scheduler,
history, best metric cùng trạng thái early stopping. Model và training config
phải khớp checkpoint; chỉ tổng số epoch được phép tăng.

## Output chính

```text
outputs/<experiment>/<timestamp>/
  checkpoints/
    best.pt
    last.pt
  artifacts/
    summary.json
    test_metrics.json
  plots/
    loss.png
    auc.png
    mrr.png
    ndcg.png
  predictions/
    test_predictions.csv
  config.yaml
  history.json
  run_info.json
  train.log
```

Validation và test báo cáo AUC, MRR, nDCG@5 và nDCG@10. News vectors được
precompute một lần mỗi evaluation pass để tránh encode lặp lại cùng news.

## Cấu hình tùy chọn

- `training.deterministic: true`: bật deterministic mode của PyTorch/CUDA.
- `loader.num_workers > 0`: worker được seed để negative sampling tái lập.
- `scheduler.type`: `none`, `reduce_on_plateau` hoặc `cosine`.
- `device`: `auto`, `cpu` hoặc `cuda`.

## Kiểm thử

```powershell
python -m pytest -q
```

Chi tiết trạng thái triển khai và các quyết định kỹ thuật nằm trong
[documents/ROADMAP.md](documents/ROADMAP.md) và
[documents/NRMS-ban-dich-tieng-viet.md](documents/NRMS-ban-dich-tieng-viet.md).