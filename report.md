# BÁO CÁO XÂY DỰNG HỆ THỐNG GỢI Ý TIN TỨC SỬ DỤNG MÔ HÌNH NRMS

## Tóm tắt

Báo cáo trình bày quá trình xây dựng và đánh giá hệ thống gợi ý tin tức dựa trên mô hình **Neural News Recommendation with Multi-Head Self-Attention (NRMS)**. Hệ thống được triển khai bằng PyTorch và thực nghiệm trên bộ dữ liệu MINDsmall. Pipeline bao gồm tiền xử lý dữ liệu, xây dựng biểu diễn tiêu đề bằng GloVe, huấn luyện mô hình, lưu và khôi phục checkpoint, đánh giá trên tập validation/test và sinh danh sách gợi ý từ lịch sử đọc.

Checkpoint tốt nhất tại epoch 8 đạt AUC 0,6878, MRR 0,3938, nDCG@5 0,3772 và nDCG@10 0,4403 trên validation. Trên test, mô hình đạt AUC 0,6442, MRR 0,3492, nDCG@5 0,3288, nDCG@10 0,3944 và HR@10 0,7340. Kết quả cho thấy mô hình học được sở thích người dùng và đưa ít nhất một bài relevant vào top 10 ở khoảng 73,40% impression test. Tuy nhiên, khoảng cách giữa validation và test vẫn cho thấy khả năng tổng quát hóa cần được cải thiện. Phiên bản hiện tại phù hợp với nghiên cứu offline; để triển khai thực tế cần bổ sung ingest bài mới, candidate retrieval, cache bền vững và xử lý cold-start.

## 1. Giới thiệu

Số lượng bài viết được xuất bản liên tục khiến người dùng khó tìm được nội dung phù hợp. Hệ thống gợi ý tin tức giải quyết vấn đề này bằng cách phân tích lịch sử tương tác và xếp hạng các bài viết theo mức độ liên quan đối với từng người dùng.

Tin tức có vòng đời ngắn, catalog thay đổi nhanh và thường xuất hiện các chủ đề, nhân vật hoặc sự kiện mới. Vì vậy, mô hình cần biểu diễn được nội dung bài viết thay vì chỉ ghi nhớ mã định danh. NRMS đáp ứng yêu cầu này bằng cách tạo news vector từ tiêu đề và tạo user vector từ lịch sử các bài đã đọc.

Mục tiêu của dự án gồm:

- xây dựng pipeline tiền xử lý MINDsmall;
- biểu diễn tiêu đề bằng embedding khởi tạo từ GloVe;
- triển khai kiến trúc NRMS bằng PyTorch;
- huấn luyện, lưu checkpoint và hỗ trợ resume;
- đánh giá bằng AUC, MRR, nDCG@5, nDCG@10 và HR@10 trên tập test;
- xuất dự đoán theo từng impression và candidate;
- cung cấp chức năng gợi ý theo lịch sử đọc.

Phiên bản hiện tại sử dụng tiêu đề làm đầu vào chính. Entity embedding và relation embedding đi kèm MINDsmall chưa được sử dụng.

## 2. Cơ sở lý thuyết

### 2.1. Biểu diễn từ

Mô hình neural xử lý tensor số thay vì chuỗi ký tự. Vì vậy, mỗi token được ánh xạ thành một word ID thông qua vocabulary, sau đó word ID được dùng để tra vector trong embedding matrix:

$$
w_i \rightarrow id_i, \qquad \mathbf{e}_i=\mathbf{E}[id_i]
$$

Trong đó $\mathbf{E}\in\mathbb{R}^{|V|\times d}$, $|V|$ là kích thước vocabulary và $d=300$ là số chiều embedding.

Hệ thống sử dụng hai token đặc biệt:

- `<PAD>` có ID 0, dùng để đưa chuỗi về độ dài cố định;
- `<UNK>` có ID 1, đại diện cho token ngoài vocabulary.

Embedding matrix được khởi tạo từ GloVe 840B 300 chiều. Chỉ các token thuộc vocabulary của dự án được lấy từ file GloVe, nhờ đó không cần nạp toàn bộ file có kích thước hơn 5,6 GB vào mô hình.

### 2.2. News encoder

News encoder nhận chuỗi embedding của các từ trong tiêu đề. Multi-head self-attention cho phép mỗi từ học quan hệ với các từ khác trong nhiều không gian biểu diễn:

$$\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})=\operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{T}}{\sqrt{d_k}}\right)\mathbf{V}$$

Đầu ra của các attention head được ghép lại. Additive attention sau đó gán trọng số cho từng từ và tổng hợp tiêu đề thành một news vector.

### 2.3. User encoder

Lịch sử đọc được biểu diễn thành chuỗi news vector. User encoder tiếp tục dùng self-attention để học quan hệ giữa các bài đã đọc và additive attention để xác định những bài phản ánh sở thích người dùng rõ nhất. Kết quả là user vector $\mathbf{u}$.

### 2.4. Tính điểm

Điểm liên quan giữa người dùng và bài ứng viên được tính bằng tích vô hướng:

$$
s(u,n)=\mathbf{u}^{T}\mathbf{n}
$$

Các bài có điểm cao hơn được xếp ở vị trí cao hơn.

### 2.5. Negative sampling

Trong mỗi impression train, bài được nhấp là positive và bài không được nhấp là negative. Mỗi mẫu huấn luyện gồm một positive và bốn negative được lấy từ cùng impression.

## 3. Dữ liệu và tiền xử lý

### 3.1. Bộ dữ liệu

Dự án sử dụng MINDsmall với hai loại dữ liệu chính:

- `news.tsv`: mã bài, category, subcategory, title, abstract và metadata;
- `behaviors.tsv`: impression, người dùng, thời gian, lịch sử đọc và ứng viên kèm nhãn click.

Thống kê dữ liệu:

| Thành phần | Số lượng |
|---|---:|
| Train behaviors | 156.965 |
| Dev behaviors | 73.152 |
| Train news | 51.282 |
| Dev news | 42.416 |
| News sau khi hợp nhất | 65.238 |

### 3.2. Chia validation và test

Tập dev được sắp xếp theo thời gian rồi chia theo tỷ lệ 50/50. Validation gồm dữ liệu từ 00:00:01 đến 09:47:21 ngày 15/11/2019; test gồm dữ liệu từ 09:47:22 đến 23:58:03 cùng ngày.

Sau khi loại impression có lịch sử rỗng:

- validation có 35.496 mẫu;
- test có 35.442 mẫu.

### 3.3. Xử lý tiêu đề

Vocabulary được xây từ title trong tập train với cấu hình chuyển chữ thường và tần suất tối thiểu bằng 1. Vocabulary có 31.055 token, bao gồm `<PAD>` và `<UNK>`.

Mỗi tiêu đề được tokenize, chuyển thành word ID, cắt ở 30 token và pad nếu ngắn hơn.

| Thống kê độ dài title | Giá trị |
|---|---:|
| Nhỏ nhất | 1 |
| Lớn nhất | 58 |
| Trung bình | 12,63 |
| Phân vị 95% | 20 |

Giới hạn 30 token bao phủ phần lớn tiêu đề và chỉ cắt một tỷ lệ nhỏ trường hợp dài.

### 3.4. Embedding

Trong 31.053 token thông thường cần tra GloVe, có 28.458 token được tìm thấy và 2.595 token không khớp. Độ bao phủ là:

$$
\frac{28.458}{31.053}\approx 91,64\%
$$

Token không khớp GloVe được khởi tạo ngẫu nhiên theo seed; vector `<PAD>` bằng 0. Embedding được cập nhật trong huấn luyện vì cấu hình `freeze_embedding` là `false`.

### 3.5. Lịch sử đọc

Mô hình giữ tối đa 50 bài gần nhất. Độ dài lịch sử trung bình xấp xỉ 32 bài, trong khi phân vị 95% khoảng 108–109 bài. Giới hạn 50 làm giảm chi phí tính toán nhưng có thể loại bỏ tín hiệu ở nhóm người dùng hoạt động nhiều.

### 3.6. Artifact

Pipeline sinh các artifact chính:

- `word_dict.pkl`;
- `word_embedding_matrix.pkl`;
- `news_title_mapping.pkl`;
- `train_samples.pkl`;
- `validation_samples.pkl`;
- `test_samples.pkl`.

Config, statistics, manifest, checksum và run metadata cũng được lưu để hỗ trợ kiểm tra tính toàn vẹn và tái lập.

## 4. Thiết kế và triển khai

### 4.1. Cấu trúc chương trình

- `src/data`: parser, text processing, embedding, preprocessing và Dataset;
- `src/models/nrms.py`: news encoder, user encoder và NRMS;
- `src/training.py`: training loop và validation;
- `src/checkpointing.py`: lưu và khôi phục checkpoint;
- `src/inference.py`: cache news vector và ranking;
- `src/reporting.py`: tổng hợp kết quả và biểu đồ;
- `scripts`: preprocess, train, evaluate và recommend;
- `tests`: kiểm thử đơn vị và tích hợp.

### 4.2. Cấu hình mô hình

| Tham số | Giá trị |
|---|---:|
| Embedding dimension | 300 |
| Attention heads | 16 |
| Head dimension | 16 |
| News/user vector dimension | 256 |
| Additive attention dimension | 200 |
| Dropout | 0,2 |
| Max history length | 50 |

### 4.3. Cấu hình huấn luyện

| Tham số | Giá trị |
|---|---:|
| Train batch size | 256 |
| Validation batch size | 512 |
| News encoding batch size | 4.096 |
| Learning rate ban đầu | 0,0002 |
| Gradient clipping | 5,0 |
| Epoch tối đa | 10 |
| Early-stopping patience | 3 |
| Metric theo dõi | nDCG@10 |
| Scheduler | ReduceLROnPlateau |
| AMP | Bật |
| Thiết bị | CUDA |

Mỗi run lưu `best.pt` và `last.pt`. Checkpoint chứa trạng thái model, optimizer, scheduler, epoch, metric tốt nhất, history và early stopping. Run chính sử dụng seed 42. Chế độ deterministic có hỗ trợ nhưng không được bật trong thực nghiệm này. Scheduler `ReduceLROnPlateau` dùng factor 0,5 và patience 1; learning rate cuối cùng là 0,00005.

### 4.4. Tối ưu đánh giá

Mỗi bài được news encoder xử lý một lần để tạo cache vector. Run chính tạo cache 65.239 entry, bao gồm bài padding, với vector 256 chiều. Evaluation dùng cache thay vì encode lặp lại một title trong nhiều impression.

## 5. Phương pháp đánh giá

### 5.1. AUC

AUC đo khả năng mô hình gán điểm cho positive cao hơn negative trong cùng impression. Giá trị càng cao càng tốt.

### 5.2. MRR

Mean Reciprocal Rank quan tâm tới vị trí của bài relevant đầu tiên:

$$
\operatorname{MRR}=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{\operatorname{rank}_i}
$$

### 5.3. nDCG@K

nDCG đánh giá thứ tự xếp hạng trong $K$ vị trí đầu và giảm trọng số của kết quả ở vị trí thấp:

$$
\operatorname{DCG@K}=\sum_{i=1}^{K}\frac{2^{rel_i}-1}{\log_2(i+1)}
$$

DCG được chuẩn hóa theo thứ tự lý tưởng để thu được nDCG từ 0 đến 1.

### 5.4. HR@10

Hit Rate tại 10 (HR@10) đo tỷ lệ impression có ít nhất một bài relevant xuất hiện trong 10 vị trí đầu:

$$
\operatorname{HR@10}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}\left(\exists j\leq 10:\operatorname{rel}_{i,j}=1\right)
$$

Trong đó $N$ là số impression và $\mathbb{I}$ là hàm chỉ thị. HR@10 phản ánh độ phủ của top 10 nhưng không phân biệt bài relevant nằm ở vị trí 1 hay vị trí 10, vì vậy cần đọc cùng MRR và nDCG.

## 6. Kết quả thực nghiệm

### 6.1. Quá trình huấn luyện

Run chính hoàn thành đủ 10 epoch với trạng thái `completed`. Thời gian huấn luyện là 1.022,67 giây, tương đương khoảng 17,04 phút; tổng thời gian của run là 1.025,19 giây. Epoch 8 là checkpoint tốt nhất theo nDCG@10.

| Epoch | Train loss | AUC | MRR | nDCG@5 | nDCG@10 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,4176 | 0,6644 | 0,3681 | 0,3512 | 0,4162 |
| 3 | 1,3032 | 0,6815 | 0,3878 | 0,3715 | 0,4347 |
| 6 | 1,2420 | 0,6858 | 0,3916 | 0,3743 | 0,4380 |
| 8 | 1,2228 | **0,6878** | **0,3938** | **0,3772** | **0,4403** |
| 10 | 1,2045 | 0,6836 | 0,3915 | 0,3744 | 0,4377 |

### 6.2. Validation và test

| Tập | AUC | MRR | nDCG@5 | nDCG@10 | HR@10 | Impressions |
|---|---:|---:|---:|---:|---:|---:|
| Validation tốt nhất (epoch 8) | 0,6878 | 0,3938 | 0,3772 | 0,4403 | — | 35.496 |
| Test | 0,6442 | 0,3492 | 0,3288 | 0,3944 | 0,7340 | 35.442 |

## 7. Chức năng gợi ý

Hệ thống nhận danh sách news ID trong lịch sử, tạo user vector và xếp hạng candidate. Nếu không truyền candidate list, hệ thống xếp hạng catalog đã preprocessing. Bài đã có trong history và candidate trùng lặp được loại bỏ.

```text
Lịch sử news ID
→ tra news vector trong cache
→ user encoder
→ tính điểm candidate
→ sắp xếp giảm dần
→ trả top-k
```

News ID phải tồn tại trong `news_title_mapping.pkl`. Chức năng hiện tại vì vậy phù hợp với thử nghiệm trên catalog MINDsmall đã biết, chưa hỗ trợ ingest trực tiếp bài mới bằng `news_id` và `title`.

## 8. Hạn chế

### 8.1. Mô hình

- Chỉ dùng title, chưa khai thác abstract, category, entity và thời gian xuất bản.
- Vocabulary theo từ khiến tên riêng hoặc sự kiện mới dễ thành `<UNK>`.
- Lịch sử giới hạn 50 bài và có thể bỏ mất tín hiệu dài hạn.
- Điểm dot product chưa xét trực tiếp freshness, popularity và diversity.
- Chưa có fallback phù hợp cho người dùng không có lịch sử.

### 8.2. Pipeline suy luận

- Cache vector chỉ tồn tại trong process và được dựng lại mỗi lần chạy CLI.
- Chưa có candidate retrieval; quét toàn catalog không phù hợp ở quy mô lớn.

### 8.3. Thực nghiệm

- Kết quả báo cáo mới dựa trên một run chính và một seed.
- Chưa đánh giá online bằng CTR, dwell time hoặc diversity.

## 9. Hướng phát triển

### 9.1. Hỗ trợ bài mới

Bổ sung pipeline ingest nhận `news_id`, title và metadata. Title được encode bằng tokenizer và news encoder hiện hành, sau đó vector được thêm vào catalog mà không cần huấn luyện lại cho từng bài.

### 9.2. Sử dụng subword tokenizer

BPE, WordPiece hoặc SentencePiece giúp giảm tỷ lệ `<UNK>` đối với tên riêng và sự kiện mới. Có thể thử pretrained language model như BERT, DistilBERT hoặc PhoBERT tùy ngôn ngữ và tài nguyên.

### 9.3. Retrieval–ranking

Hệ thống production nên có hai tầng:

1. retrieval chọn nhanh một tập ứng viên nhỏ từ catalog lớn;
2. NRMS xếp hạng lại ứng viên theo sở thích người dùng.

Retrieval có thể kết hợp vector index, thời gian xuất bản, category, ngôn ngữ và business rules.

### 9.4. Bổ sung đặc trưng

Nên nghiên cứu abstract, category, entity, thời gian xuất bản, popularity, xu hướng, ngữ cảnh truy cập và diversity trong danh sách kết quả.

### 9.5. Nâng cao thực nghiệm

- chạy nhiều seed và báo cáo trung bình cùng độ lệch chuẩn;
- đánh giá theo độ dài history và tỷ lệ OOV;
- đo latency, memory và throughput bên cạnh ranking metrics.

## 11. Kết luận

Dự án đã xây dựng thành công pipeline NRMS đầy đủ cho thực nghiệm offline trên MINDsmall, từ preprocessing, training, checkpoint, validation, test đến recommendation. Kiến trúc word embedding, multi-head self-attention và additive attention phù hợp với nguyên lý NRMS. Việc precompute news vector giúp giảm chi phí đánh giá lặp lại.

Checkpoint tốt nhất tại epoch 8 đạt nDCG@10 0,4403 trên validation. Trên 35.442 impression test, mô hình đạt AUC 0,6442, MRR 0,3492, nDCG@5 0,3288, nDCG@10 0,3944 và HR@10 0,7340. Mô hình có khả năng đưa bài relevant vào top 10 ở phần lớn impression, nhưng khoảng cách validation–test cho thấy cần thêm thí nghiệm để cải thiện khả năng tổng quát hóa và vị trí xếp hạng của bài relevant.