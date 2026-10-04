# DeepWeeds — Lab Day 2

## Notebook

- Kaggle notebook: **[DÁN LINK NOTEBOOK KAGGLE SAU KHI TẠO]**
- Chạy notebook `code/lab_day2.ipynb` theo thứ tự từ trên xuống. Notebook đọc bộ dữ liệu đã gắn vào Kaggle Input:
  - `/kaggle/input/datasets/quchuy2k4/images`
  - `/kaggle/input/datasets/quchuy2k4/files-csv/file-csv`
- Đẩy code trong `submissions/2A202602929_vu_quoc_huy/code/` lên fork GitHub trước khi chạy. Cell đầu clone fork vào `/kaggle/working/lab_repo` một lần; notebook lấy code từ đúng commit đó. Dùng **Add Input** cho dataset ảnh và CSV đã có, không cần tạo thêm Kaggle Dataset cho code.
- Bật GPU và Internet trong Kaggle. Internet dùng để cài `timm`/`thop` và tải trọng số ImageNet lần đầu.
- Notebook dùng code và lưu kết quả trong `/kaggle/working/lab_repo/submissions/2A202602929_vu_quoc_huy`. Nếu cần chạy phiên bản code mới trong cùng session, khởi động lại session hoặc cập nhật bản clone trước khi chạy lại cell đầu.

## Lộ trình chạy

1. Cài thư viện, ghi phiên bản môi trường, xác nhận GPU và đường dẫn dữ liệu.
2. Chạy EDA và kiểm tra fold 0: số ảnh, giao giữa các split, ảnh thiếu, phân bố lớp, ảnh mẫu.
3. Chạy kiểm tra pipeline: batch/nhãn, focal loss với `gamma=0`, Mixup/CutMix, overfit batch nhỏ và gộp Conv-BN.
4. So sánh năm backbone; chọn backbone và recipe chỉ bằng validation.
5. So sánh các phương pháp inference, hiệu chuẩn nhiệt độ trên validation, đo độ trễ có warmup và đồng bộ GPU.
6. Xem bảng/đồ thị validation, chốt cấu hình rồi đổi `RUN_FINAL_TEST = False` thành `True` trong cell final. Chỉ sau đó mới chạy final và baseline cho seed 0, 1, 2. Cell kiểm tra cấu hình để không chạy lại test đã có.
7. Chạy `eval.py score`, `eval.py grade`, tạo `results.xlsx`, biểu đồ và ảnh phân tích lỗi.

Các lần train lưu `last.pt` sau mỗi epoch để tiếp tục nếu phiên Kaggle bị ngắt. Checkpoint bị loại khỏi Git bởi `.gitignore`; không commit dataset hoặc checkpoint. Không thay cấu hình sau khi đã tạo dự đoán test. Nếu cell final báo cấu hình cũ không khớp, giữ lại kết quả đã ghi và dùng một `exp_id` mới cho một lần chạy được phép theo yêu cầu môn học.

## Đầu ra

- `predictions/`: CSV theo định dạng `eval.py`, gồm validation, final test và dự đoán chưa hiệu chuẩn để so sánh ECE.
- `curves/`: đường cong train/validation của từng backbone, recipe và seed final.
- `results.xlsx`: bảy sheet kết quả theo rubric.
- `report_assets/`: phân bố lớp, ảnh mẫu, đồ thị accuracy-latency, confusion matrix và gallery lỗi.
- `runs/`: cấu hình, log theo epoch, summary, logits validation và checkpoint cục bộ.
- `report.md`: dàn ý báo cáo; chỉ điền số sau khi notebook chạy xong.

## Môi trường

Notebook ghi các phiên bản Python, PyTorch, torchvision, timm, NumPy, pandas và GPU thật vào `environment.json`. Các con số kết quả và phần cứng trong báo cáo phải lấy từ lần chạy Kaggle thực tế.

Sau khi chạy xong, tải `submission_artifacts.zip` từ Output của Kaggle, giải nén vào **gốc repo** để lấy predictions, curves, workbook và hình báo cáo; zip không chứa checkpoint. Điền `report.md`, thêm link Kaggle notebook ở đầu README này, rồi commit các artifact cần nộp vào fork.
