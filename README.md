# EvolutionaryComputation.20222
**Bài tập lớn môn Tính toán tiến hóa HUST kỳ 20222**

Giải bài toán xác định hướng quay cho các cảm biến trong mạng cảm biến với vị trí của các cảm biến và mục tiêu được khởi tạo ngẫu nhiên, các cảm biến có số lượng góc quay cố định.
Mô hình bài toán tham khảo bài báo "Maximizing heterogeneous coverage in over and under provisioned visual sensor networks" (./doc/AlZishan2018.pdf).

Cài đặt giải thuật tham lam SOGA và các tiến hóa:
- GA (Drive: https://drive.google.com/drive/folders/1bGgQF6wOV_t7RNAsevTvm9S5VAV7IRHP)
- PSO/ DPSO
- LSHADE

Cấu trúc thư mục như sau:
 - doc: tài liệu môn học, project
 - images
 - saved: dữ liệu và lời giải của các thuật toán, các thư mục 'continuos' sử dụng dữ liệu từ bài toán cảm biến có hướng quay liên tục, còn lại sinh dữ liệu theo bài báo
 - Các file .py: cài đặt các lớp, hàm làm việc với dữ liệu, đánh giá kết quả, thuật toán (đã có chú thích trong mã nguồn)
 - Các file .ipynb: tạo dữ liệu, thử nghiệm thuật toán, đánh giá.
 - Tham khảo tóm tắt bài báo (Paper summarize) và thuyết trình (Presentation)

### Tiến độ công việc
- [x] Cài đặt mô hình bài toán và thuật toán LSHADE
- [x] Cài đặt thuật toán GA
- [x] Cài đặt thuật toán PSO và DPSO
- [x] Cài đặt thuật toán LSHADE
- [ ] Áp dụng transfer function cho thuật toán trên miền liên tục (PSO, LSHADE) để giải bài toán trên miền rời rạc

