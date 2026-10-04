# Báo cáo Lab Day 2 — DeepWeeds

> Hoàn thiện các ô `[điền ...]` bằng kết quả thực tế trong `results.xlsx`, `eval_out/` và `environment.json`. Không ghi kết quả ước lượng. Ghi rõ nếu đã giảm batch size, số epoch hoặc số thí nghiệm vì giới hạn GPU.

## 1. Tóm tắt

- Bài toán: phân loại ảnh DeepWeeds thành 9 lớp.
- Thiết lập: fold `[điền]`, số seed cuối `[điền]`, GPU `[điền]`.
- Backbone/recipe/inference cuối: `[điền]`.
- Macro-F1 test mean ± std: `[điền]`; top-1 test mean ± std: `[điền]`.
- Chênh lệch macro-F1 so với `T00 + I00`: `[điền]`.
- Kết luận chính dựa trên kết quả: `[điền]`.

## 2. Dữ liệu và thiết lập

Đã dùng nguyên các split fold 0: `train_subset0.csv`, `val_subset0.csv` và `test_subset0.csv`. Kiểm tra cho thấy train `[điền]`, validation `[điền]`, test `[điền]` ảnh; giao giữa các split `[điền]`; ảnh thiếu `[điền]`.

Tỉ lệ lớp train lớn nhất/nhỏ nhất là `[điền]`. Ảnh mẫu có kích thước `[điền]` và mode `[điền]`. Đính kèm `report_assets/class_distribution.png` và `report_assets/sample_images.png`.

Thiết lập train: input `[điền]`, batch `[điền]`, optimizer `[điền]`, learning rate backbone/head `[điền]`, weight decay `[điền]`, loss `[điền]`, augmentation `[điền]`, AMP `[điền]`, epoch `[điền]`. Phiên bản thư viện và GPU lấy từ `environment.json`.

## 3. So sánh backbone

Chèn bảng từ sheet `Backbones` của `results.xlsx`. Thảo luận macro-F1 validation cùng số tham số, GMAC, thời gian train/epoch và latency batch 1. Nêu backbone được chọn và lý do dựa trên validation/chi phí.

Đính kèm `report_assets/backbone_accuracy_latency.png`. Nhận xét về tốc độ hội tụ, quá khớp và khác biệt giữa GMAC với latency: `[điền từ đường cong và phép đo]`.

## 4. Ablation công thức huấn luyện

Chèn bảng `Training`. Mỗi so sánh thay đổi một yếu tố so với `T00`. Báo cáo delta macro-F1 và các lớp hiếm nếu có.

| Yếu tố | Cấu hình | Macro-F1 val | Δ so với T00 | Nhận xét |
|---|---|---:|---:|---|
| Khởi tạo | `[điền]` | `[điền]` | `[điền]` | `[điền]` |
| Augmentation | `[điền]` | `[điền]` | `[điền]` | `[điền]` |
| Loss | `[điền]` | `[điền]` | `[điền]` | `[điền]` |
| Kết hợp | `[điền]` | `[điền]` | `[điền]` | `[điền]` |

Diễn giải khác biệt nhỏ hơn độ lệch chuẩn là chưa phân biệt được; các sweep này chỉ có một seed.

## 5. Phương pháp inference và latency

Chèn sheet `Inference` và `Latency`. Đoạn benchmark gồm warmup, ít nhất 50 lượt đo và đồng bộ GPU trước/sau. Latency hiện ghi là thời gian forward model, chưa tính đọc ảnh/tiền xử lý; xem điều kiện trong notebook trước khi so với yêu cầu triển khai.

Đính kèm `report_assets/inference_accuracy_latency.png`. So sánh 1-view, flip TTA, five-crop TTA, gộp xác suất/logit và temperature scaling. Sau khi chốt phương pháp inference bằng validation, mỗi seed final fit nhiệt độ riêng trên validation của chính seed đó rồi áp dụng nguyên giá trị lên test. Ghi các giá trị T và ECE validation trước/sau: `[điền]`.

## 6. Kết quả cuối trên test

Chỉ dùng kết quả sau khi đã chốt cấu hình bằng validation. Chèn sheet `Final` và `PerClass`; báo cáo mean ± std qua các seed cho final và mốc `T00 + I00`.

- Final macro-F1 test: `[điền]`; mốc: `[điền]`; delta: `[điền]`.
- Final top-1 test: `[điền]`.
- ECE trước/sau hiệu chuẩn: `[điền]` / `[điền]`.
- Recall Chinee Apple: `[điền]`; Snake Weed: `[điền]`.
- `eval.py grade`: `[điền tổng điểm và ghi chú]`.

Đính kèm `report_assets/confusion_F01_test.png`. Lớp/cặp lớp nhầm nhiều nhất: `[điền]`. Quan sát gallery lỗi `report_assets/error_gallery_F01_seed0.png`; giả thuyết về lỗi thị giác hoặc nhãn: `[điền]`.

## 7. Kết luận và khuyến nghị triển khai

1. Cấu hình tốt nhất và cải thiện so với mốc: `[điền số liệu]`.
2. Yếu tố đóng góp nhiều nhất trong backbone, recipe và inference: `[điền, dẫn số liệu]`.
3. Cấu hình có p95 batch-1 không quá 100 ms (nếu có), GPU và điều kiện đo: `[điền]`.
4. Khuyến nghị nếu giới hạn thời gian thực là 30–100 ms/frame: `[điền]`.

## 8. Hạn chế và việc tiếp theo

Nêu số seed và fold, các thí nghiệm đã giảm/bỏ do thời gian GPU, độ lệch giữa validation/test, hạn chế của split ngẫu nhiên không theo địa điểm, và khả năng lệch mùa/miền ảnh: `[điền]`.

## 9. Tái lập

- Repo fork: `[điền link repo]`
- Kaggle notebook: `[điền link notebook]`
- Thứ tự chạy: xem `README.md` trong thư mục này.
- Danh sách `exp_id` và cấu hình: `results.xlsx` và `runs/<exp_id>/seed<k>/config.json`.
