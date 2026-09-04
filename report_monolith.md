# BÁO CÁO THỰC TẬP
## Tìm hiểu về Monolith — Real Time Recommendation System With Collisionless Embedding Table

**Nguồn paper:** Zhuoran Liu et al., ByteDance Inc. — ORSUM@ACM RecSys 2022 ([arXiv:2209.07663](https://arxiv.org/abs/2209.07663))

---

## 1. Tổng quan

### 1.1. Monolith giải quyết vấn đề gì?

Monolith là hệ thống Recommendation System (RecSys) quy mô production do ByteDance phát triển, nhằm giải quyết **hai vấn đề cốt lõi** mà các framework Deep Learning đa dụng (TensorFlow, PyTorch) chưa đáp ứng được khi áp dụng vào bài toán gợi ý:

| Vấn đề | Mô tả ngắn gọn |
|---|---|
| **Sparse & Dynamic Features** | Dữ liệu RecSys chủ yếu là categorical, thưa, và liên tục thay đổi (user/item mới xuất hiện hàng ngày). Các framework truyền thống dùng biến dense kích thước cố định → không phù hợp. |
| **Concept Drift** | Sở thích người dùng thay đổi rất nhanh (trong vài phút), trong khi pipeline batch training truyền thống tách biệt hoàn toàn giai đoạn train và serve → model luôn bị "cũ". |

### 1.2. Đóng góp chính của paper

Paper đề xuất ba đóng góp:

1. **Collisionless Embedding Table** — Bảng embedding không va chạm dựa trên Cuckoo HashMap, kèm cơ chế lọc tần suất (frequency filtering) và embedding có thời hạn (expirable embedding) để tiết kiệm bộ nhớ.
2. **Online Training Architecture** — Kiến trúc huấn luyện trực tuyến sẵn sàng cho production, cho phép model cập nhật từ phản hồi người dùng theo thời gian thực (mức phút).
3. **Fault Tolerance Trade-off** — Chứng minh rằng có thể đánh đổi một phần độ tin cậy hệ thống (giảm tần suất snapshot) để lấy khả năng học thời gian thực mà không ảnh hưởng đáng kể đến chất lượng model.

---

## 2. Bối cảnh và động lực

### 2.1. Tại sao RecSys khác bài toán Deep Learning thông thường?

Dữ liệu RecSys có hai đặc điểm khác biệt cơ bản so với Computer Vision hay NLP:

**Đặc điểm 1: Features sparse, categorical và dynamic**

- **Sparse:** Một training sample chỉ liên quan đến 1 user trong hàng trăm triệu user. Nếu biểu diễn one-hot, vector gần như toàn số 0.
- **Categorical:** `user_id = 13527` không có nghĩa user này "lớn hơn" user 4509. Các giá trị chỉ là nhãn danh mục.
- **Dynamic:** User mới đăng ký, video mới được upload liên tục → feature space không cố định, embedding table phải grow theo thời gian.

**Đặc điểm 2: Non-stationary distribution (Concept Drift)**

Trong ML truyền thống, ta giả định $P_{train}(X,Y) \approx P_{future}(X,Y)$. Nhưng với RecSys, sở thích user có thể thay đổi chỉ trong vài phút (ví dụ: sáng xem bóng đá, chiều chuyển sang tìm hiểu AI). Model train trên dữ liệu cũ sẽ nhanh chóng trở nên lỗi thời.

### 2.2. Hạn chế của các framework hiện tại

| Hạn chế | Giải thích |
|---|---|
| Fixed-size embedding | TensorFlow/PyTorch dùng `Variable` kích thước cố định → không thể mở rộng khi có ID mới |
| Hash collision | Kỹ thuật hash-trick giảm bộ nhớ nhưng gây va chạm ID → nhiều user/item chia sẻ cùng embedding → suy giảm chất lượng |
| Batch-only pipeline | Train và serve tách biệt hoàn toàn → model không thể phản ứng với phản hồi real-time |

---

## 3. Thiết kế hệ thống Monolith

### 3.1. Kiến trúc tổng thể

Monolith tuân theo mô hình **Worker — Parameter Server (PS)** phân tán:

```
┌──────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│   Training   │     │   Training PS        │     │   Serving PS     │
│   Worker     │────>│  (Dense + Sparse     │────>│  (Phục vụ user   │
│  (Tính toán  │     │   Parameters)        │     │   real-time)     │
│   gradient)  │     │                      │     │                  │
└──────────────┘     └──────────────────────┘     └──────────────────┘
       ^                                                   │
       │              Feedback Loop                        │
       └───────────────────────────────────────────────────┘
```

- **Worker:** Thực hiện forward/backward pass, tính gradient.
- **Training PS:** Lưu trữ và cập nhật tham số (cả dense lẫn sparse).
- **Serving PS:** Phục vụ model cho user, nhận tham số đồng bộ từ Training PS.

Tham số được chia thành hai nhóm:
- **Dense parameters:** Trọng số của mạng neural sâu (DNN weights).
- **Sparse parameters:** Bảng embedding tương ứng với các đặc trưng categorical.

### 3.2. Collisionless Hash Table

Đây là đóng góp kỹ thuật quan trọng nhất của paper.

**Vấn đề với hash-trick truyền thống:**

Giả sử không gian ID gốc là $2^{48}$, hash-trick ánh xạ xuống không gian nhỏ hơn (ví dụ $2^{25}$). Khi nhiều ID khác nhau bị ánh xạ vào cùng một slot → **collision** → các user/item khác nhau chia sẻ cùng embedding → model không phân biệt được.

**Giải pháp của Monolith — Cuckoo Hashing:**

Paper sử dụng Cuckoo HashMap làm nền tảng cho bảng embedding:

- Duy trì **hai bảng** $T_0$, $T_1$ với hai hàm băm khác nhau $h_0(x)$, $h_1(x)$.
- Khi chèn phần tử A vào $T_0$ tại vị trí $h_0(A)$:
  - Nếu vị trí trống → chèn trực tiếp.
  - Nếu đã bị chiếm bởi B → đẩy B ra, thử chèn B vào $T_1$ tại $h_1(B)$.
  - Lặp lại cho đến khi ổn định hoặc rehash.
- Độ phức tạp: **O(1)** cho lookup/delete (worst-case), **O(1) amortized** cho insert.
- **Không có collision** — mỗi ID có embedding riêng.

**Tối ưu bộ nhớ:**

Bảng embedding không va chạm sẽ rất lớn nếu chèn mọi ID. Paper đề xuất hai cơ chế:

| Cơ chế | Mô tả |
|---|---|
| **Frequency Filtering** | Chỉ chèn ID vào bảng khi số lần xuất hiện vượt ngưỡng. ID hiếm (long-tail) bị huấn luyện thiếu, loại bỏ không ảnh hưởng chất lượng. |
| **Expirable Embedding** | Gắn timestamp cho mỗi ID, tự động xóa sau khi không hoạt động trong thời gian định trước. User không còn active hoặc video lỗi thời sẽ bị loại bỏ. |

### 3.3. Online Training

Quá trình huấn luyện chia thành **hai giai đoạn**:

**Giai đoạn 1 — Batch Training:**
- Huấn luyện trên dữ liệu lịch sử từ HDFS.
- Chỉ chạy **một lượt** (one pass) qua dữ liệu.
- Dùng khi thay đổi kiến trúc model và cần train lại từ đầu.

**Giai đoạn 2 — Online Training:**
- Sau khi deploy, model **không ngừng học**.
- Training Worker sử dụng dữ liệu real-time từ Kafka.
- Training PS đồng bộ tham số sang Serving PS theo khoảng thời gian **mức phút**.
- Model phản ứng với phản hồi user gần như tức thì.

```
User interaction → Kafka (user actions) ───┐
                                           ▼
Features → Kafka (features) ──────> Flink Online Joiner
                                           │
                                           ▼
                                   Kafka (training examples)
                                      │           │
                                      ▼           ▼
                              Online Training   HDFS dump
                              (real-time)       (for batch)
```

### 3.4. Streaming Engine & Online Joiner

**Streaming Engine** gồm ba hàng đợi Kafka:
1. **User action log:** Ghi lại hành động (click, like, mua hàng...).
2. **Features:** Các đặc trưng tại thời điểm serving.
3. **Training examples:** Kết quả ghép nối, sẵn sàng cho training.

**Online Joiner** (tác vụ Flink) ghép features với user actions:
- Dùng **unique request key** để ghép cặp chính xác.
- Xử lý **delayed feedback** (user có thể mua hàng vài ngày sau khi xem): dùng cache trong RAM + kho key-value trên đĩa.
- Áp dụng **negative sampling** để xử lý mất cân bằng positive/negative, kèm **log-odds correction** khi serving để đảm bảo ước lượng không thiên lệch.

### 3.5. Parameter Synchronization

Paper khai thác ba đặc điểm quan trọng để thiết kế cơ chế đồng bộ hiệu quả:

- Sparse parameters chiếm phần lớn kích thước model -> Tập trung tối ưu đồng bộ sparse
- Trong thời gian ngắn, chỉ một tập con nhỏ ID được cập nhật -> Chỉ đồng bộ **incremental** (touched keys), không cần gửi toàn bộ
- Dense parameters thay đổi chậm hơn nhiều so với sparse -> Đồng bộ dense ít thường xuyên hơn

**Kết quả thiết kế:**
- **Sparse sync:** Mức phút, chỉ gửi embedding của các ID đã được cập nhật (touched keys). Ví dụ: 100.000 ID × 1024 chiều × 4 bytes = ~400MB/phút — rất nhẹ.
- **Dense sync:** Mức ngày, lên lịch vào lúc traffic thấp nhất (nửa đêm).
- Cập nhật **on-the-fly** — Serving PS không cần dừng phục vụ.

### 3.6. Fault Tolerance

- **Snapshot** toàn bộ Training PS **mỗi ngày** (thay vì thường xuyên hơn).
- Nếu một PS gặp sự cố → khôi phục từ snapshot ngày trước → mất tối đa 1 ngày cập nhật.
- Paper chứng minh điều này chấp nhận được qua phép tính:
  - 1000 PS, tỷ lệ lỗi 0,01%/máy/ngày → 1 PS lỗi mỗi 10 ngày.
  - 15 triệu DAU phân bố đều → mất feedback 1 ngày của ~15.000 user (0,1% DAU).
  - Với dense parameters thay đổi chậm, mất 1 ngày trên 1/1000 PS là không đáng kể.

→ **Trade-off:** Giảm tần suất snapshot → tiết kiệm đáng kể chi phí tính toán mà gần như không ảnh hưởng chất lượng model.

---

## 4. Tổng kết

Monolith giải quyết hai vấn đề cốt lõi của RecSys production bằng hai thiết kế chính:

```
Vấn đề 1: Sparse + Dynamic Features
    └──> Giải pháp: Collisionless Embedding Table (Cuckoo HashMap)
         + Frequency Filtering + Expirable Embedding

Vấn đề 2: Concept Drift
    └──> Giải pháp: Online Training Architecture
         + Streaming Engine (Kafka + Flink)
         + Incremental Parameter Sync (mức phút cho sparse, mức ngày cho dense)
         + Fault Tolerance Trade-off (snapshot mỗi ngày)
```

Paper cho thấy cả hai thiết kế đều mang lại cải thiện trong production: collisionless embedding tăng AUC ổn định, online training cải thiện AUC 14–18% so với batch training trong A/B test thực tế.

---