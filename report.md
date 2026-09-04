# BÁO CÁO XÂY DỰNG HỆ THỐNG GỢI Ý TIN TỨC SỬ DỤNG MÔ HÌNH NRMS

## Mục lục

- [Tóm tắt](#tóm-tắt)
- [1. Giới thiệu](#1-giới-thiệu)
- [2. Cơ sở lý thuyết](#2-cơ-sở-lý-thuyết)
	- [2.1. Biểu diễn từ](#21-biểu-diễn-từ)
	- [2.2. News encoder](#22-news-encoder)
	- [2.3. User encoder](#23-user-encoder)
	- [2.4. Tính điểm](#24-tính-điểm)
	- [2.5. Negative sampling](#25-negative-sampling)
- [3. Dữ liệu và tiền xử lý](#3-dữ-liệu-và-tiền-xử-lý)
	- [3.1. Bộ dữ liệu](#31-bộ-dữ-liệu)
	- [3.2. Chia validation và test](#32-chia-validation-và-test)
	- [3.3. Xử lý tiêu đề](#33-xử-lý-tiêu-đề)
	- [3.4. Embedding](#34-embedding)
	- [3.5. Lịch sử đọc](#35-lịch-sử-đọc)
	- [3.6. Artifact](#36-artifact)
- [4. Thiết kế và triển khai](#4-thiết-kế-và-triển-khai)
	- [4.1. Cấu trúc chương trình](#41-cấu-trúc-chương-trình)
	- [4.2. Cấu hình mô hình](#42-cấu-hình-mô-hình)
	- [4.3. Cấu hình huấn luyện](#43-cấu-hình-huấn-luyện)
	- [4.4. Tối ưu đánh giá](#44-tối-ưu-đánh-giá)
- [5. Phương pháp đánh giá](#5-phương-pháp-đánh-giá)
	- [5.1. AUC](#51-auc)
	- [5.2. MRR](#52-mrr)
	- [5.3. nDCG@K](#53-ndcgk)
	- [5.4. HR@10](#54-hr10)
- [6. Kết quả thực nghiệm](#6-kết-quả-thực-nghiệm)
	- [6.1. Quá trình huấn luyện](#61-quá-trình-huấn-luyện)
	- [6.2. Giai đoạn 1: khảo sát biến thể với seed 42](#62-giai-đoạn-1-khảo-sát-biến-thể-với-seed-42)
	- [6.3. Giai đoạn 2: đánh giá độ ổn định trên năm seed](#63-giai-đoạn-2-đánh-giá-độ-ổn-định-trên-năm-seed)
- [7. Chức năng gợi ý](#7-chức-năng-gợi-ý)
- [8. Hạn chế](#8-hạn-chế)
	- [8.1. Mô hình](#81-mô-hình)
	- [8.2. Pipeline suy luận](#82-pipeline-suy-luận)
	- [8.3. Thực nghiệm](#83-thực-nghiệm)
- [9. Hướng phát triển](#9-hướng-phát-triển)
	- [9.1. Hỗ trợ bài mới](#91-hỗ-trợ-bài-mới)
	- [9.2. Sử dụng subword tokenizer](#92-sử-dụng-subword-tokenizer)
	- [9.3. Retrieval–ranking](#93-retrievalranking)
	- [9.4. Bổ sung đặc trưng](#94-bổ-sung-đặc-trưng)
	- [9.5. Nâng cao thực nghiệm](#95-nâng-cao-thực-nghiệm)
- [10. Kết luận](#10-kết-luận)

## Tóm tắt

Báo cáo trình bày quá trình xây dựng và đánh giá hệ thống gợi ý tin tức dựa trên mô hình **Neural News Recommendation with Multi-Head Self-Attention (NRMS)**. Hệ thống được triển khai bằng PyTorch và thực nghiệm trên bộ dữ liệu MINDsmall. Pipeline bao gồm tiền xử lý dữ liệu, xây dựng biểu diễn tiêu đề bằng GloVe, huấn luyện mô hình, lưu và khôi phục checkpoint, đánh giá trên tập validation/test và sinh danh sách gợi ý từ lịch sử đọc.

Thực nghiệm được tiến hành theo hai giai đoạn. Giai đoạn thứ nhất dùng seed 42 để khảo sát các biến thể của additive attention dimension, attention head dimension và tỷ lệ negative sampling; từ đó chọn hai cấu hình có additive attention dimension 200 và 256. Giai đoạn thứ hai đánh giá độ ổn định của hai cấu hình này trên năm seed 42–46. Trung bình trên năm seed, cấu hình 256 đạt validation nDCG@10 $0,441972\pm0,001178$ và test nDCG@10 $0,391790\pm0,001999$, cao hơn cấu hình 200 lần lượt là $0,439426\pm0,001999$ và $0,391001\pm0,003094$. Cấu hình 256 có giá trị trung bình cao hơn trên toàn bộ metric validation và test, đồng thời có độ lệch chuẩn thấp hơn trên toàn bộ test metric. Kết quả cho thấy additive attention dimension 256 nhỉnh hơn và ổn định hơn trong phạm vi năm seed được khảo sát.

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

Trong mỗi impression train, bài được nhấp là positive và bài không được nhấp là negative. Cấu hình chính dùng một positive và bốn negative lấy từ cùng impression. Một thí nghiệm ablation tăng số negative lên tám để đánh giá ảnh hưởng của tỷ lệ negative sampling.

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

| Tham số | Hai cấu hình đánh giá độ ổn định | Biến thể khác ở giai đoạn khảo sát |
|---|---:|---:|
| Embedding dimension | 300 | — |
| Attention heads | 16 | — |
| Head dimension | 16 | 32 |
| News/user vector dimension | 256 | 512 |
| Additive attention dimension | 200 và 256 | 300 |
| Dropout | 0,2 | — |
| Max history length | 50 | — |

### 4.3. Cấu hình huấn luyện

| Tham số | Giá trị |
|---|---:|
| Train batch size | 256 |
| Validation batch size | 512 |
| News encoding batch size | 4.096 |
| Learning rate ban đầu | 0,0002 |
| Gradient clipping | 5,0 |
| Weight decay | 0 |
| Epoch tối đa | 15 |
| Early-stopping patience | 3 |
| Minimum delta | 0,0001 |
| Metric theo dõi | nDCG@10 |
| Scheduler | ReduceLROnPlateau |
| AMP | Bật |
| Thiết bị | CUDA |

Mỗi run lưu `best.pt` và `last.pt`. Checkpoint chứa trạng thái model, optimizer, scheduler, epoch, metric tốt nhất, history và early stopping. Chế độ deterministic có hỗ trợ nhưng không được bật; độ ổn định được đánh giá bằng cách thay seed từ 42 đến 46. Scheduler `ReduceLROnPlateau` dùng factor 0,5, patience 1 và learning rate tối thiểu $10^{-6}$. Tùy diễn biến validation, learning rate cuối cùng của các run là 0,0001 hoặc 0,00005.

Run seed 42 của cấu hình additive dimension 200 được tái sử dụng từ giai đoạn khảo sát và có giới hạn 10 epoch; các run bổ sung có giới hạn 15 epoch. Checkpoint tốt nhất của run seed 42 xuất hiện tại epoch 8 nên vẫn nằm trong giới hạn huấn luyện. Khác biệt này được giữ lại khi tổng hợp kết quả và cần được xem là một hạn chế nhỏ của thiết kế thực nghiệm.

### 4.4. Tối ưu đánh giá

Mỗi bài được news encoder xử lý một lần để tạo cache gồm 65.239 entry, bao gồm bài padding, với vector 256 chiều. Evaluation dùng cache thay vì encode lặp lại một title trong nhiều impression.

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

> Chi tiết ở file [results.xlsx](./result.xlsx)

### 6.1. Quá trình huấn luyện

Thực nghiệm gồm hai giai đoạn. Trước hết, các biến thể được so sánh với cùng seed 42 nhằm hạn chế ảnh hưởng của khởi tạo ngẫu nhiên trong bước sàng lọc. Hai cấu hình additive attention dimension 200 và 256 cho kết quả tốt nhất nên được giữ lại. Sau đó, mỗi cấu hình được đánh giá trên năm seed 42–46 để đo hiệu năng trung bình và độ ổn định.

### 6.2. Giai đoạn 1: khảo sát biến thể với seed 42

| Biến thể | Seed | Best epoch | Val nDCG@10 | Test AUC | Test MRR | Test nDCG@5 | Test nDCG@10 | HR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Additive dim 200 | 42 | 8 | 0,4403 | 0,6442 | 0,3492 | 0,3288 | **0,3944** | **0,7340** |
| Additive dim 256 | 42 | 8 | **0,4428** | **0,6465** | **0,3499** | **0,3307** | 0,3941 | 0,7311 |
| Additive dim 300 | 42 | 8 | 0,4396 | 0,6464 | 0,3473 | 0,3283 | 0,3927 | 0,7320 |
| Head dim 32, additive dim 200 | 42 | 8 | 0,4403 | 0,6430 | 0,3424 | 0,3229 | 0,3881 | 0,7285 |
| Negative sampling ratio 8 | 42 | 8 | 0,4362 | 0,6450 | 0,3463 | 0,3264 | 0,3916 | 0,7302 |

Ở seed 42, additive dimension 256 đạt validation nDCG@10, test AUC, MRR và nDCG@5 cao nhất; additive dimension 200 đạt test nDCG@10 và HR@10 cao nhất. Hai cấu hình vì vậy được chọn để đánh giá tiếp trên nhiều seed. Additive dimension 300, head dimension 32 và negative sampling ratio 8 không tạo ra cải thiện đủ nhất quán để đi tiếp.

### 6.3. Giai đoạn 2: đánh giá độ ổn định trên năm seed

Kết quả được trình bày dưới dạng trung bình $\pm$ độ lệch chuẩn mẫu trên các seed 42–46, với $n=5$ và `ddof=1`. Metric validation được lấy tại checkpoint tốt nhất theo validation nDCG@10 của từng run.

| Additive attention dim | Validation AUC | Validation MRR | Validation nDCG@5 | Validation nDCG@10 |
|---:|---:|---:|---:|---:|
| 200 | 0,687798 ± 0,002925 | 0,392951 ± 0,001634 | 0,376307 ± 0,001355 | 0,439426 ± 0,001999 |
| **256** | **0,688054 ± 0,001766** | **0,397281 ± 0,000905** | **0,379605 ± 0,001597** | **0,441972 ± 0,001178** |

| Additive attention dim | Test AUC | Test MRR | Test nDCG@5 | Test nDCG@10 |
|---:|---:|---:|---:|---:|
| 200 | 0,644965 ± 0,003066 | 0,346076 ± 0,003171 | 0,325947 ± 0,003703 | 0,391001 ± 0,003094 |
| **256** | **0,645993 ± 0,002403** | **0,347000 ± 0,002244** | **0,327256 ± 0,003024** | **0,391790 ± 0,001999** |

Additive dimension 256 có trung bình cao hơn trên cả bốn validation metric và cả năm test metric. Mức tăng test còn nhỏ: AUC tăng 0,001028, MRR tăng 0,000924, nDCG@5 tăng 0,001309, nDCG@10 tăng 0,000789 và HR@10 tăng 0,000203. Tuy nhiên, cấu hình 256 cũng có độ lệch chuẩn thấp hơn trên toàn bộ test metric, đặc biệt nDCG@10 giảm từ 0,003094 xuống 0,001999. Vì vậy, kết luận phù hợp là additive attention dimension 256 đạt hiệu năng trung bình nhỉnh hơn và nhìn chung ổn định hơn additive dimension 200; chưa thể khẳng định khác biệt có ý nghĩa thống kê nếu chưa thực hiện kiểm định ghép cặp theo seed.

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

- Hai cấu hình được chọn đã được đánh giá trên cùng năm seed, nhưng cỡ mẫu năm seed vẫn nhỏ và chưa có kiểm định ý nghĩa thống kê.
- Run seed 42 của additive dimension 200 dùng giới hạn 10 epoch, trong khi các run bổ sung dùng giới hạn 15 epoch; checkpoint tốt nhất ở epoch 8 nên ảnh hưởng dự kiến nhỏ nhưng quy trình chưa hoàn toàn đồng nhất.
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

- mở rộng số seed ngoài 42–46 và giữ toàn bộ điều kiện huấn luyện đồng nhất;
- bổ sung khoảng tin cậy và kiểm định ghép cặp theo seed hoặc theo từng impression;
- đánh giá theo độ dài history và tỷ lệ OOV;
- đo latency, memory và throughput bên cạnh ranking metrics.

## 10. Kết luận

Dự án đã xây dựng thành công pipeline NRMS đầy đủ cho thực nghiệm offline trên MINDsmall, từ preprocessing, training, checkpoint, validation, test đến recommendation. Kiến trúc word embedding, multi-head self-attention và additive attention phù hợp với nguyên lý NRMS. Việc precompute news vector giúp giảm chi phí đánh giá lặp lại.

Thiết kế thực nghiệm hai giai đoạn đã sàng lọc các biến thể bằng seed 42, sau đó đánh giá độ ổn định của hai cấu hình additive attention dimension 200 và 256 trên năm seed 42–46. Cách tổng hợp này tránh kết luận dựa trên một run đơn lẻ và phản ánh rõ hơn ảnh hưởng của khởi tạo ngẫu nhiên.

Additive dimension 256 đạt validation nDCG@10 trung bình $0,441972\pm0,001178$ và test nDCG@10 $0,391790\pm0,001999$, so với $0,439426\pm0,001999$ và $0,391001\pm0,003094$ của additive dimension 200. Cấu hình 256 có trung bình cao hơn trên toàn bộ metric được đánh giá và độ lệch chuẩn thấp hơn trên toàn bộ test metric. Do đó, additive attention dimension 256 là lựa chọn phù hợp hơn trong hai cấu hình, dù mức chênh lệch test còn nhỏ và cần được xác nhận bằng thêm seed hoặc kiểm định thống kê.