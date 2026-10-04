# DeepWeeds — Lab Day 2

## Notebook

- Kaggle notebook: **[Lab Day 2 — Vu Quoc Huy](https://www.kaggle.com/code/quchuy2k4/lab-day2-vuquochuy-2a202602929)**
- Chạy notebook `code/lab_day2.ipynb` theo thứ tự từ trên xuống. Notebook đọc bộ dữ liệu đã gắn vào Kaggle Input:
  - `/kaggle/input/datasets/quchuy2k4/images`
  - `/kaggle/input/datasets/quchuy2k4/files-csv/file-csv`
- Đẩy code trong `submissions/2A202602929_vu_quoc_huy/code/` lên fork GitHub trước khi chạy. Cell đầu clone fork vào `/kaggle/working/lab_repo` một lần; notebook lấy code từ đúng commit đó. Dùng **Add Input** cho dataset ảnh và CSV đã có, không cần tạo thêm Kaggle Dataset cho code.
- Bật GPU và Internet trong Kaggle. Internet dùng để cài `timm`/`thop` và tải trọng số ImageNet lần đầu.
- Notebook dùng code và lưu kết quả trong `/kaggle/working/lab_repo/submissions/2A202602929_vu_quoc_huy`. Nếu cần chạy phiên bản code mới trong cùng session, khởi động lại session hoặc cập nhật bản clone trước khi chạy lại cell đầu.

## Lộ trình chạy

1. Cài thư viện, ghi phiên bản môi trường, xác nhận GPU và đường dẫn dữ liệu.
2. Chạy EDA và kiểm tra fold 0: số ảnh, giao giữa các split, ảnh thiếu, phân bố lớp, ảnh mẫu.
3. Chạy kiểm tra pipeline: batch/nhãn, focal loss với `gamma=0`, Mixup/CutMix, overfit batch nhỏ, gộp Conv-BN; kiểm tra một optimizer step ở đúng batch size và tải trước pretrained weights cho cả năm backbone. Nếu bước này báo OOM, giảm `BATCH_SIZE` xuống 32 **trước** lượt train dài đầu tiên.
4. So sánh năm backbone với 15 epoch, cùng seed/split/recipe; chọn backbone bằng macro-F1 validation và chi phí. Chạy T00–T07: khởi tạo, CutMix, color jitter, focal loss, class-balanced CE (`beta=0.999`) và EMA (`decay=0.995`) mỗi lần chỉ đổi một yếu tố. T08 kết hợp hai yếu tố chọn bằng validation.
5. So sánh các phương pháp inference, hiệu chuẩn nhiệt độ trên validation, đo độ trễ có warmup và đồng bộ GPU.
6. Xem bảng/đồ thị validation, chốt cấu hình rồi đổi `RUN_FINAL_TEST = False` thành `True` trong cell final. Chỉ sau đó mới chạy final và baseline với **20 epoch** cho seed 0, 1, 2. Mỗi final seed fit một nhiệt độ trên validation của chính seed đó rồi áp dụng nguyên giá trị cho test. Cell kiểm tra cấu hình để không chạy lại test đã có.
7. Chạy `eval.py score`, `eval.py grade`, tạo `results.xlsx`, biểu đồ và ảnh phân tích lỗi.

Để đo thời gian trước khi dùng hết quota, ở Bước 1 có thể tạm đặt `BACKBONES = ["resnet50"]` và chạy riêng B01. Xem `mean_epoch_seconds`, sau đó khôi phục `BACKBONES = list(BACKBONE_CANDIDATES)` và chạy lại cell: B01 đã hoàn thành sẽ được dùng lại nếu cấu hình giữ nguyên. Sau khi đủ năm backbone, cell in ước lượng thô giờ GPU còn cần cho ablation và chung kết. Nếu ngân sách không đủ, điều chỉnh `ABLATION_EPOCHS` và `FINAL_EPOCHS` **trước khi bắt đầu các lượt tương ứng**; giữ cùng epoch giữa các lượt cùng nhóm và giữa final/baseline.

Các lần train lưu `last.pt` sau mỗi epoch. Sau Bước 1, Bước 2 và Bước 4, notebook tạo `/kaggle/working/lab_resume.zip` gồm checkpoint và kết quả trung gian; ZIP chỉ giữ optimizer checkpoint của lượt chưa hoàn thành để tiết kiệm dung lượng. Nếu phải ngắt một cell train giữa chừng, chạy `save_resume_archive()` trong cell mới trước khi kết thúc phiên. Tải hoặc lưu output này trước khi đổi phiên Kaggle. Ở phiên mới, gắn ZIP qua Add Input và đặt `RESUME_ARCHIVE` trong cell đầu bằng đường dẫn ZIP; cell đầu kiểm tra mã băm code và phiên bản thư viện rồi phục hồi các lượt đã chạy. `lab_resume.zip` chỉ dùng để tiếp tục thí nghiệm, **không commit vào Git**. Không thay cấu hình sau khi đã tạo dự đoán test. Nếu cell final báo cấu hình cũ không khớp, giữ lại kết quả đã ghi và dùng một `exp_id` mới cho một lần chạy được phép theo yêu cầu môn học.

## Đầu ra

- `predictions/`: CSV theo định dạng `eval.py`, gồm validation, final test và dự đoán chưa hiệu chuẩn để so sánh ECE.
- `curves/`: đường cong train/validation của từng backbone, recipe và seed final.
- `results.xlsx`: bảy sheet kết quả theo rubric.
- `report_assets/`: phân bố lớp, ảnh mẫu, đồ thị accuracy-latency, confusion matrix và gallery lỗi.
- `runs/`: notebook giữ cấu hình, log theo epoch, summary, logits validation và checkpoint trong Kaggle working directory; các file này không nằm trong artifact commit.
- `report.md`: báo cáo đã điền bằng kết quả của lần chạy Kaggle bên dưới.

## Môi trường

Lần chạy đã nộp dùng seed 0, 1, 2 cho final và baseline; năm backbone và các recipe sàng lọc dùng seed 0. Môi trường thực tế trong `environment.json`: Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.29, NumPy 2.1.3, pandas 2.3.3, Tesla T4 ×2 (train trên cuda:0). Các kết quả và phần cứng trong báo cáo được lấy từ lần chạy Kaggle này.

Với lần chạy đã hoàn thành, các file trong `submission_artifacts.zip` đã được giải nén vào thư mục submission, báo cáo đã điền, link Kaggle đã thêm và artifact đã commit lên fork. ZIP kết quả không chứa checkpoint, logits hoặc thư mục `runs/`; `lab_resume.zip` được giữ riêng để tiếp tục/tái lập, không commit vào fork.
